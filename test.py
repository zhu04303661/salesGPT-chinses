import os
import openai
openai.organization = "org-NAEqVRfHzXYmnaBokxcqAQ1J"
openai.api_key = os.getenv("OPENAI_API_KEY")
x=openai.Model.list()
print x
