from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from pydantic import BaseModel,Field
from typing import TypedDict,Annotated,Optional,Literal
load_dotenv()
# good to apply data validation
model=ChatMistralAI()
# schema
json_schema={
  "title": "Review",
  "type": "object",
  "properties": {
    "key_themes": {
      "title": "Key Themes",
      "description": "Write down all the key themes discussed in the review in a list",
      "type": "array",
      "items": {
        "type": "string"
      }
    },
    "summary": {
      "title": "Summary",
      "description": "A brief summary of the review",
      "type": "string"
    },
    "sentiment": {
      "title": "Sentiment",
      "description": "Return the sentiment of the review as either positive or negative",
      "enum": [
        "+ve",
        "-ve"
      ],
      "type": "string"
    },
    "pros": {
      "title": "Pros",
      "description": "List the good features of the product mentioned in the review",
      "anyOf": [
        {
          "type": "array",
          "items": {
            "type": "string"
          }
        },
        {
          "type": "null"
        }
      ],
      "default": "null"
    },
    "cons": {
      "title": "Cons",
      "description": "List all the cons mentioned in the review",
      "anyOf": [
        {
          "type": "array",
          "items": {
            "type": "string"
          }
        },
        {
          "type": "null"
        }
      ],
      "default": "null"
    }
  },
  "required": [
    "key_themes",
    "summary",
    "sentiment"
  ]
}

structured_model=model.with_structured_output(json_schema) # return response in structured format

result=structured_model.invoke("""The hardware is great, but the software feels bloated. There are too many pre-installed apps that I can’t remove. Also, One UI looks outdated compared to other brands. Hoping for a software update to fix this.

I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it’s an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes everything lightning fast—whether I’m gaming, multitasking, or editing photos. The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.

The S-Pen integration is a great touch for note-taking and quick sketches, though I don’t use it often. What really blows me away is the 200MP camera—the night mode is stunning, capturing crisp, vibrant images even in low light. Zooming up to 10x actually works well for distant objects, but anything beyond 30x loses quality.

However, the weight and size make it a bit uncomfortable for one-handed use. Also, Samsung’s One UI still comes with bloatware—why do I need five different Samsung apps for things Google already provides? The $1,300 price tag is also a hard pill to swallow.

Pros:
Insanely powerful processor (great for gaming and productivity)
Stunning 200MP camera with incredible zoom capabilities
Long battery life with fast charging
S-Pen support is unique and useful

Cons:
Bulky and heavy—not great for one-handed use
Bloatware still exists in One UI
Expensive compared to competitors"""

)
print(result)