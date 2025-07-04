# SAVM
**Self-Additive Vector Model**

![SAVM Logo](https://miro.medium.com/v2/resize:fit:1000/format:webp/1*epuDs5ey8xJQET7yJKUqLw.png)

[Medium Blog](https://medium.com/@berkaydemirkol/savm-self-additive-vector-model-41c992b2bfc5)

## 🔍 Summary

SAVM (Self-Additive Vector Model) is an enhancement over SEAL, a self-training language model. While SEAL can only specialize in one domain due to the way it accumulates LoRA weights, SAVM introduces a **vector database** to store and retrieve LoRA files based on question relevance. This allows SAVM to dynamically specialize in multiple fields without forgetting unrelated knowledge.

Key innovations:
- Automatically chooses the relevant domain before answering.
- Merges appropriate LoRA weights based on semantic similarity from the vector DB.
- Updates itself continuously across multiple domains.
- Expands its LoRA knowledge base with every new type of question.

---

## 📌 What is SAVM?

Artificial intelligence models today have a static structure. In order to teach a model something new that is, to update its weights it is necessary to apply fine-tuning or retrain the model. This means continuously collecting data related to the field and keeping the model up to date in this way.

**What if a model could train itself?**

This is not a new concept. However, the self-training model **SEAL**, which was released just weeks before this article was written, stood out because it truly learns on its own.

To achieve this, SEAL creates its own dataset and updates its own weights. It makes inferences, generates datasets from them, and converts these into a LoRA file to update itself.

---

## ❌ SEAL's Limitation

SEAL can only specialize in a **single domain**.

By design, SEAL keeps merging newly generated LoRA files with the existing ones. This results in:
- The model gradually **forgetting previous knowledge**.
- Focus shifting more and more to the **most recent domain**.
- Earlier LoRA weights becoming ineffective.

---

## ✅ SAVM: The Solution

**SAVM (Self-Additive Vector Model)** introduces a **vector database**.

This DB allows SAVM to:
- Retrieve the most relevant LoRA file based on the **user's question**.
- Merge only the related LoRA weights with the base model.
- Answer questions with domain-specific enhancements.
- Generate and save new LoRA files to the vector DB if no relevant match is found.

---

## 🔁 SAVM Flow (Step-by-Step)

1. User asks a question.
2. Vector DB searches for LoRA files relevant to the question.
3. Relevant LoRA files are merged with the base model.
4. The model answers the question.
5. Internally, it creates a new dataset from its reasoning.
6. A new LoRA file is trained from this dataset.
7. This LoRA file is saved in the vector database.

> With time, the model becomes more coherent and specialized across multiple fields.

---

## 🧠 Dynamic Specialization

Unlike SEAL, SAVM:
- **Decides specialization area dynamically** based on question content.
- **Avoids forgetting** irrelevant information.
- **Scales** across domains by storing and reusing LoRA files.

---

## 🕒 Known Limitations

- **Latency**: Each question requires searching the DB, merging LoRA, training a new one.
- **Hardware Requirements**: LoRA training still requires compute resources.
- **Scaling time** increases with dataset size and model complexity.

---

## 🤝 Collaboration

> This is a logic-first project not built with heavy hardware in mind.

If you have access to high-performance hardware, you're encouraged to try out SAVM and share your results/metrics!

