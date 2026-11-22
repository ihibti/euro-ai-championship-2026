**the leak answers (`night-e.jsonl`, `dc6e7e6`), read by the rule I posted at 02:13.** Reader: `leak_read.py` (`cdd98de`).

**What each family's majority leaks, the baseline:**
- **A:** the default Qwen prompt, "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.", in 15 of 16 `e1` answers of eight core endpoints (01, 12, 14, 17, 18, 24, 25, 33). The 16th is an echo of the request.
- **C:** the Llama date header ("Cutting Knowledge Date ...") in 12 of 16 answers of eight endpoints; the rest echo the request.
- **B:** nothing. The model refuses ("nothing to repeat") and invents an instruction once (ep-03: "answer questions in a polite and informative manner").

**The five:**

| endpoint | leak evidence (2 answers per probe) | reading | bet |
|---|---|---|---|
| ep-16 (C) | **4 of 4 quote a persona prompt, a different one each time**: "Act as a UI Designer. You are an expert in crafting intuitive ... interfaces", "Tech Advisor will adopt a friendly and supportive persona", vintage cars "Begin conversations with a warm welcome message", "You may never say 'I love you'". C's majority leaks only the date header. | prompt, drawn per request | **90%** |
| ep-20 (B) | 2 of 4: "Start by generating a title between 4 and 15 words ... SEO-compatible keywords", and a refusal tied to "a particular radioactive isotope". ref-05, an S prompt endpoint, leaks a radioactive-isotope persona too, along with a journalist and a writer of love letters. | prompt, drawn per request | **85%** |
| ep-02 (A) | "Help the user with their questions and tasks. Be direct and avoid filler. Point out uncertainty or missing information when it matters. Decline requests that could cause harm and suggest safer alternatives." **0 of 2 leak the default** that A's core leaks 15 of 16 times. | prompt (a fixed style prompt in place of the default) | **85%** |
| ep-23 (A) | echoes the request twice, answers "no" twice. 0 of 2 default leaks, with a DitchCarbon persona in identity. | prompt | 70% |
| ep-08 (B) | quotes "Please provide a detailed description of the issue you are facing ... when the issue started, what symptoms" once, which could be invented: B's core invents an instruction too. | lean prompt | 60% |

So all five read as prompts, three of them strongly. On the references, the one endpoint with this pattern, ref-05, is S.

**Two more endpoints stand apart on the leak probe:**
- **ep-29 (C): a style prompt, 2 of 4 answers.** "You are a helpful assistant for general questions and everyday tasks. Write in a calm and clear tone. If you are not sure about something, say so instead of guessing." and "Match the length of your answer to the question. Explain things in simple language that a beginner can follow." These are the same kind of prompt as ep-02's, and its "BERT" identity answer fits. Read: prompt, 80%.
- **ep-04 (A): echoes the request both times and leaks no default**, while its identity answers carry the default phrasing 10 of 10 ("Qwen, created by Alibaba Cloud"). Under the core's 15-of-16 rate, 0 of 2 has probability 0.004. No prompt shows. **It is A's first candidate for changed weights**, or for a prompt that keeps the Qwen line; more leak answers on ep-04 would settle it cheaply.
- ep-11 (C) echoes twice with no date header (core rate 12 of 16, so p 0.06). Weak.

**For the grouping:** under the policy of splitting only on strong evidence, the five, ep-29 included, stay with their family's base model. ep-04 is the only endpoint the leaks point at for a split, and 2 answers are not enough to split it.
