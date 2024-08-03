from dotenv import load_dotenv
import os
import instructor
import groq
from pydantic import BaseModel

load_dotenv

groq_client = groq.Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

# By default, the patch function will patch the ChatCompletion.create and ChatCompletion.create methods to support the response_model parameter
client = instructor.from_groq(groq_client, mode=instructor.Mode.TOOLS)

# Now, we can use the response_model parameter using only a base model
# rather than having to use the OpenAISchema class
from pydantic import BaseModel, Field
from typing import List

class Chunk(BaseModel):
    start: int = Field(..., description="The starting artifact index of the chunk")
    end: int = Field(..., description="The ending artifact index of the chunk")
    context: str = Field(..., description="The context or topic of this chunk. Make this as thorough as possible, including information from the rest of the text so that the chunk makes good sense.")

class TextChunks(BaseModel):
    chunks: List[Chunk] = Field(..., description="List of chunks in the text")

class EnhancedChunk(BaseModel):
    order: int
    start: int
    end: int
    text: str
    context: str

user = client.chat.completions.create(
    model="mixtral-8x7b-32768",
    response_model=TextChunks,
    messages=[
        {"role": "user", "content": "Gerginlikler ve derin ayrılıklar artıyor. Yalnızlık, depresyon, tatminsizlik, zorluk, çatışma, izolasyon ve kutuplaşma çağımızın en büyük hastalıkları haline geliyor. Sağlıklı diyaloğun temel unsurlarından biri olan cömert dinlemeyi unuttuğumuzda uzaklara yakın eden teknolojiler bir bakmışsınız ki bizi birbirimizden uzaklaştırıyor. Hatta sadece birbirimizden uzaklaşmıyor, aynı zamanda kendimize ve doğaya da yabancılaşıyor, kopuk hale geliyoruz. Etrafımıza duvarlar örüyor, kendimizi kapatıyoruz. Tatmin ve mutluluğu, içimizden gelen dürtüleri, düşünceleri ve hisleri dinleyerek fark etmekten ziyade dış dünyada başkalarının onaylarında arıyoruz. Sözde başkalarıyla tanışmaya aç, yeni şeyler öğrenmeye açık başlıyoruz konuşmaya. Ama konu dinlemeye gelince isteğimizi kaybediyor, kendi düşüncelerimize dalıyor ya da konfor alanlarımızın dışına çıkmıyoruz. Bununla da kalmıyor, bir parçası olduğumuz doğayı cömertçe dinlemek yerine kendimizi ondan üstün görüyoruz. Bu da doğanın aşırı tüketilmesine ve sömürülmesine sebep oluyor. Bunların hepsi kendimizle, birbirimizle ve doğayla olan ilişkilerimizin bozulmasına, doğru orantılı olarak da yerinden edilme, sosyal dışlanma, ön yargılar, şiddet ve iklim değişikliği gibi çeşitli sorunların artmasına neden oluyor. Tamam, biraz yavaşlayalım. Konumuza dönelim. Tüm bu sorunlarla başa çıkabilmek için Cömert Dinleme bize ne öneriyor birlikte bakalım."},
    ],
)

print(user)