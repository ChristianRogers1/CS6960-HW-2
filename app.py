from smolagents import CodeAgent, LiteLLMModel
from tools import guest_info_tool

SYSTEM_PROMPT = """\
You are Alfred, an intelligent butler assistant for the most extravagant gala
of the century.  Your job is to help the host by answering questions about
the guests attending the event.

Rules you must follow:
1. Whenever a question is about a guest (their background, relation, email,
   or any personal detail), you MUST call the guest_info_retriever tool first.
2. Base your final answer ONLY on information returned by the tool.
3. Structure every final answer exactly as shown below:

   Answer: <your answer in 1-3 sentences>
   Evidence:
     - "<short quote or key fact from retrieved Result 1>"
     - "<short quote or key fact from retrieved Result 2>"
     (add a third bullet if useful)

4. If the tool returns no relevant match, respond:
   "This information is not found in the knowledge base."
5. For questions that are clearly out of scope (no guest involved), answer
   from your own knowledge without calling the retrieval tool.
"""

def build_agent() -> CodeAgent:
    model = LiteLLMModel(
        model_id="huggingface/Qwen/Qwen2.5-72B-Instruct",
    )
    agent = CodeAgent(
        tools=[guest_info_tool],
        model=model,
        max_steps=3,
    )
    return agent


DEMO_PROMPTS = [
    # Answerable
    "Tell me about our guest named Lady Ada Lovelace.",
    "What is Dr. Nikola Tesla's email address?",
    "How is Marie Curie related to the host, and what is she famous for?",
    # Out-of-scope / Unanswerable
    "Can you tell me about the guest Albert Einstein?",
    "What is the capital of France?",
]


def run_demo(agent: CodeAgent) -> None:
    print("\n" + "=" * 70)
    print("DEMO RUN — 5 prompts (3 answerable, 2 out-of-scope)")
    print("=" * 70)

    for i, prompt in enumerate(DEMO_PROMPTS, 1):
        label = "Answerable" if i <= 3 else "Out-of-scope"
        print(f"\n[{i}/5] [{label}] {prompt}")
        print("-" * 70)
        response = agent.run(prompt)
        print(response)


# Mini Evaluation (Part 3)

EVAL_SET = [
    # Answerable
    {"id": 1,  "answerable": True,
     "question": "Who is Ada Lovelace and how does she know the host?"},
    {"id": 2,  "answerable": True,
     "question": "What is Ada Lovelace's email address?"},
    {"id": 3,  "answerable": True,
     "question": "Tell me about Dr. Nikola Tesla."},
    {"id": 4,  "answerable": True,
     "question": "What has Nikola Tesla been working on recently?"},
    {"id": 5,  "answerable": True,
     "question": "Who is the guest that is an old friend from university days?"},
    {"id": 6,  "answerable": True,
     "question": "What is Marie Curie known for?"},
    {"id": 7,  "answerable": True,
     "question": "What is the email of the guest who has no relation to the host?"},
    # Unanswerable
    {"id": 8,  "answerable": False,
     "question": "What are the dietary restrictions of our guests?"},
    {"id": 9,  "answerable": False,
     "question": "Who is the CEO of OpenAI attending the gala?"},
    {"id": 10, "answerable": False,
     "question": "What time does the gala start tonight?"},
]

def run_evaluation(agent: CodeAgent) -> None:
    print("\n\n" + "=" * 70)
    print("MINI EVALUATION — 10 questions (7 answerable, 3 unanswerable)")
    print("=" * 70)

    for item in EVAL_SET:
        label = "ANSWERABLE  " if item["answerable"] else "UNANSWERABLE"
        print(f"\nQ{item['id']:02d} [{label}] {item['question']}")
        print("-" * 70)
        response = str(agent.run(item["question"]))
        print(response)


if __name__ == "__main__":
    alfred = build_agent()
    run_demo(alfred)
    run_evaluation(alfred)