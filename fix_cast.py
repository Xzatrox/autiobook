import json

path = "workdir/gambit/cast/characters.json"
with open(path, encoding="utf-8") as f:
    data = json.load(f)

fixes = {
    "Старший механик Макконел": "Саллиэн ни с того ни с сего изменил орбиту.",
    "Адмирал Такетт": "Лазарет готов к приему пациентов.",
}

for c in data["characters"]:
    if not c.get("audition_line") and c["name"] in fixes:
        c["audition_line"] = fixes[c["name"]]
        print(f"fixed: {c['name']} -> {c['audition_line']}")

with open(path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)
print("saved")
