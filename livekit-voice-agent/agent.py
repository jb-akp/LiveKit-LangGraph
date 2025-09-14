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
    bey
)
from livekit.plugins.turn_detector.multilingual import MultilingualModel

from rag_graph import create_workflow  # <-- our compiled LangGraph app

load_dotenv(".env.local")

class InterviewAgent(Agent):
    def __init__(self) -> None:
        super().__init__(instructions=(
            "You are a professional interviewer conducting a job interview. "
            "The LangGraph workflow will drive the conversation flow. "
            "Simply speak the questions and responses as they come from the graph. "
            "Be conversational, professional, and helpful throughout the interview process."
        ))

async def entrypoint(ctx: agents.JobContext):
    # 1) Build/compile the LangGraph app (Runnable)
    interview_workflow = create_workflow()

    # 2) Wrap it as an LLM for LiveKit via the LangChain plugin
    #    (LLMAdapter knows how to drive LangGraph workflows as an LLM stream)
    lg_llm = langchain.LLMAdapter(graph=interview_workflow)

    # 3) Configure the rest of the realtime pipeline
    session = AgentSession(
        stt=deepgram.STT(model="nova-3", language="multi"),
        llm=lg_llm,  # <-- use the adapter here instead of openai.LLM(...)
        tts=cartesia.TTS(model="sonic-2", voice="f786b574-daa5-4673-aa0c-cbe3e8534c02"),
        vad=silero.VAD.load(),
        turn_detection=MultilingualModel(),
    )

    avatar = bey.AvatarSession(
        avatar_id="694c83e2-8895-4a98-bd16-56332ca3f449",  # ID of the Beyond Presence avatar to use
    )

    # Start the avatar and wait for it to join
    await avatar.start(session, room=ctx.room)

    await session.start(
        room=ctx.room,
        agent=InterviewAgent(),
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    # Start the interview workflow - the graph will drive the conversation
    print("Starting interview workflow...")
    # The graph will automatically begin with the first question

if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
