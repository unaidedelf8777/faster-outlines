# credit: the examples in this are coming from the official outlines repo readme.
import outlines
from faster_outlines import patch

patch(outlines)
schema = '''{
    "title": "Character",
    "type": "object",
    "properties": {
        "name": {
            "title": "Name",
            "maxLength": 10,
            "type": "string"
        },
        "age": {
            "title": "Age",
            "type": "integer"
        },
        "armor": {"$ref": "#/definitions/Armor"},
        "weapon": {"$ref": "#/definitions/Weapon"},
        "strength": {
            "title": "Strength",
            "type": "integer"
        }
    },
    "required": ["name", "age", "armor", "weapon", "strength"],
    "definitions": {
        "Armor": {
            "title": "Armor",
            "description": "An enumeration.",
            "enum": ["leather", "chainmail", "plate"],
            "type": "string"
        },
        "Weapon": {
            "title": "Weapon",
            "description": "An enumeration.",
            "enum": ["sword", "axe", "mace", "spear", "bow", "crossbow"],
            "type": "string"
        }
    }
}'''

model = outlines.models.transformers("mistralai/Mistral-7B-Instruct-v0.2", device="cuda:0", model_kwargs={"load_in_8bit": True})
print("Model loaded.")
generator = outlines.generate.json(model, schema)
character = generator("Give me a character description")
print(character)

prompt = """You are a sentiment-labelling assistant.
Is the following review positive or negative?

Review: This restaurant is just awesome! i love it!
"""

generator = outlines.generate.choice(model, ["Positive", "Negative"])
answer = generator(prompt)
print(f"--------\n{prompt}\n\n{answer}\n--------")

prompt = "sqrt(2)="
generator = outlines.generate.format(model, float)
answer = generator(prompt, max_tokens=10)
print(f"--------\n{prompt}\n\n{answer}\n--------")

prompt = "What is the IP address of the Google DNS servers? "

generator = outlines.generate.text(model)

generator = outlines.generate.regex(
    model,
    r"((25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(25[0-5]|2[0-4]\d|[01]?\d\d?)",
)
answer = generator(prompt, max_tokens=30)

print(f"--------\n{prompt}\n\n{answer}\n--------")
