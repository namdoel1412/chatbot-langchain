"""Main entrypoint for the app."""
import logging
import pickle
import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.templating import Jinja2Templates
from langchain.vectorstores import VectorStore

from callback import QuestionGenCallbackHandler, StreamingLLMCallbackHandler
from query_data import get_chain
from schemas import ChatResponse
from langchain.vectorstores import FAISS, Chroma
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings

app = FastAPI()
templates = Jinja2Templates(directory="templates")
vectorstore: Optional[VectorStore] = None
gmo_retriever = None

# expired
#os.environ["OPENAI_API_KEY"] = "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"
#os.environ["OPENAI_API_KEY"] = "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"
#os.environ["OPENAI_API_KEY"] = "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"
#os.environ["OPENAI_API_KEY"] = "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"
#os.environ["OPENAI_API_KEY"] = "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"
#os.environ["OPENAI_API_KEY"] = "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"

# active
#os.environ["OPENAI_API_KEY"] = "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"
#os.environ["OPENAI_API_KEY"] = "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"
#os.environ["OPENAI_API_KEY"] = "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"
#os.environ["OPENAI_API_KEY"] = "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"
#os.environ["OPENAI_API_KEY"] = "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"
os.environ["OPENAI_API_KEY"] = "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"

@app.on_event("startup")
async def startup_event():
    # logging.info("loading vectorstore")
    # if not Path("vectorstore.pkl").exists():
    #     raise ValueError("vectorstore.pkl does not exist, please run ingest.py first")
    # with open("vectorstore.pkl", "rb") as f:
    #     global vectorstore
    #     vectorstore = pickle.load(f)
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    gmoDB = Chroma(persist_directory=os.path.join('db', 'gmo'), embedding_function=embeddings)
    global gmo_retriever, vectorstore
    gmo_retriever = gmoDB.as_retriever(search_kwargs={"k": 2})
    vectorstore = gmo_retriever


@app.get("/")
async def get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.websocket("/chat")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    question_handler = QuestionGenCallbackHandler(websocket)
    stream_handler = StreamingLLMCallbackHandler(websocket)
    chat_history = []
    qa_chain = get_chain(vectorstore, question_handler, stream_handler)
    # Use the below line instead of the above line to enable tracing
    # Ensure `langchain-server` is running
    # qa_chain = get_chain(vectorstore, question_handler, stream_handler, tracing=True)
    #prompt_ext = """. Language: Vietnamese. Require: You must List the results with each line separately. Please write an introduction for your answer"""
    prompt_ext = """"""
    prompt_suffix = """.You must List the reply with each line separately. Please write an introduction for your answer.Trả lời bằng tiếng Việt hoặc dịch câu trả lời sang Tiếng Việt"""
    flag = 0
    while True:
        try:
            # Receive and send back the client message
            question = await websocket.receive_text()
            resp = ChatResponse(sender="you", message=question, type="stream")
            await websocket.send_json(resp.dict())

            # Construct a response
            start_resp = ChatResponse(sender="bot", message="", type="start")
            await websocket.send_json(start_resp.dict())
            query = question + prompt_ext + prompt_suffix
            # prompt_ext = ""
            result = await qa_chain.acall(
                {"question": query, "chat_history": chat_history}
            )
            print(result)
            # chat_history.append((query, result["answer"]))

            end_resp = ChatResponse(sender="bot", message="", type="end")
            await websocket.send_json(end_resp.dict())
        except WebSocketDisconnect:
            logging.info("websocket disconnect")
            break
        except Exception as e:
            logging.error(e)
            resp = ChatResponse(
                sender="bot",
                message="Sorry, something went wrong. Try again.",
                type="error",
            )
            await websocket.send_json(resp.dict())


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="10.16.31.39", port=9000)
