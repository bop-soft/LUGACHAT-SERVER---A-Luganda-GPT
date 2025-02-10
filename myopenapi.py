from openai import OpenAI
import os

def call_openai(transcription):
    try:
      open_api_key = os.getenv("OPEN_API_KEY")
      client = OpenAI(
        api_key=open_api_key
      )

      completion = client.chat.completions.create(
        model="gpt-4o-mini",
        store=True,
        messages=[
          {"role": "user", "content": "The following is a question in Luganda. Make sure to also reply back in Luganda." + transcription}
        ]
      )

      print(completion.choices[0].message)
      return completion.choices[0].message
    except Exception as e:
      print(f"OpenAI API error: {e}")
      return "Wabaddewo obuzibu ne Open API. Kakasa nti wajisasulidde!"
