import os
import json
question_dict = []
count = 1
small_files = os.listdir(f"selected_stories/small/")
for f in small_files:
    _file = open(f"selected_stories/small/{f}","r")
    _data = _file.readlines()
    content = ""
    for x in _data:
        content += str(x).strip()
    question = {
        "question_id": count,
        "text": "Summarize the given story in less than 150 words while preserving high coherence, consistency, fluency, and relevance.\n\n" + content,
        "category": "small"
    }
    question_dict.append(question)
    count += 1

mid_files = os.listdir(f"selected_stories/mid/")
for f in mid_files:
    _file = open(f"selected_stories/mid/{f}","r")
    _data = _file.readlines()
    content = ""
    for x in _data:
        content += str(x).strip()
    question = {
        "question_id": count,
        "text": "Summarize the given news in less than 150 words while preserving high coherence, consistency, fluency, and relevance.\n\n" + content,
        "category": "mid"
    }
    question_dict.append(question)
    count += 1

large_files = os.listdir(f"selected_stories/large/")
for f in large_files:
    _file = open(f"selected_stories/large/{f}","r")
    _data = _file.readlines()
    content = ""
    for x in _data:
        content += str(x).strip()
    question = {
        "question_id": count,
        "text": "Summarize the given news in less than 150 words while preserving high coherence, consistency, fluency, and relevance.\n\n" + content,
        "category": "large"
    }
    question_dict.append(question)
    count += 1

print(len(question_dict))   
questions = [json.dumps(q) for q in question_dict]
with open("./table/new_question.jsonl", "w") as fo:   
    fo.write("\n".join(questions))