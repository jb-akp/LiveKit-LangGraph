# agent.py
from dotenv import load_dotenv
from livekit import agents
from livekit.agents import AgentSession, Agent, RoomInputOptions
from livekit.plugins import (
    langchain,   # <-- this is key
    cartesia,
    deepgram,
    noise_cancellation,
    silero,
)
from livekit.plugins.turn_detector.multilingual import MultilingualModel

from rag_graph import create_workflow  # <-- our compiled LangGraph app

load_dotenv()

class RAGAssistant(Agent):
    def __init__(self) -> None:
        super().__init__(instructions=(
            "You are a helpful voice RAG assistant. "
            "Answer questions about the 2024 stock performance PDF. "
            "If you need to look things up, use the retrieval tool."
        ))

async def entrypoint(ctx: agents.JobContext):
    # 1) Build/compile the LangGraph app (Runnable)

    # 2) Wrap it as an LLM for LiveKit via the LangChain plugin
    #    (LLMAdapter knows how to drive LangGraph workflows as an LLM stream)
    lg_llm = langchain.LLMAdapter(graph=create_workflow())

    # 3) Configure the rest of the realtime pipeline
    session = AgentSession(
        stt=deepgram.STT(model="nova-3", language="multi"),
        llm=lg_llm,  # <-- use the adapter here instead of openai.LLM(...)
        tts=cartesia.TTS(model="sonic-2", voice="f786b574-daa5-4673-aa0c-cbe3e8534c02"),
        vad=silero.VAD.load(),
        turn_detection=MultilingualModel(),
    )

    await session.start(
        room=ctx.room,
        agent=RAGAssistant(),
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    # Optional: greet immediately
    await session.generate_reply(instructions="Say hello and explain you can answer questions about the 2024 stock PDF.")

if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
