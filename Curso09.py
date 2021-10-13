import tweepy as tw

#Essas keys devem ser geradas
consumer_key = 'SbvqnVGTI7gx4B5L8RMU3nakH'
consumer_secret = 'BcSGYicI0Ubf5mayzpLMVkRDfVPbSrYLPP3zeMvM6i6874tEoB'
access_token = '917548352446791685-hfmY6pFMPDiW59VGUFlXbIUUbp8MKxu'
access_token_secret = 'n00UiYneBr6KGYYY51sPyrBv9uwSFYbWwoJWmzabO9ica'

auth = tw.OAuthHandler(consumer_key, consumer_secret)

auth.set_access_token(access_token,access_token_secret)

api = tw.API(auth)

tweet = api.update_status("Python e Twitter na #Alura")

tweet._json

tweets = tw.Cursor(api.search,
         q="data science",
         since='2019-01-01',
         lang='pt').items(10)

for tweeet in tweets:
  print(tweet.text)

tweets = tw.Cursor(api.search,
         q="python",
         since='2019-01-01',
         lang='pt').items(20)


for tweet in tweets:
      print(tweet.user.screen_name)
      print(tweet.user.location)
      print('-----')
      print('\n')

famosos = ['cauareymond','aguiarthur','ivetesangalo','ClaudiaLeitte','neymarjr','BruMarquezine','mariruybarbosa',
'FePaesLeme','Tatawerneck','FlaviaAleReal','julianapaes','dedesecco','SabrinaSato','ahickmann','gusttavo_lima','Anitta',
'CarolCastroReal','gio_antonelli','maisa','cleooficial','gewbank','taisdeverdade','otaviano','bernardipaloma',
'IngridGuimaraes','olazaroramos','GalisteuOficial','debranascimento','FioMattheis','moalfradique','Nandacostareal']

for famoso in famosos:
    tweets = tw.Cursor(api.search,
                       q=famoso,
                       since='2019-01-01').items(20)
    print('Autor: ', famoso)
    print('Imagens postadas:')

    for tweet in tweets:
        if 'media' in tweet.entities:
            print(tweet.entities['media'][0]['media_url'])

    print('-----')
    print('\n')

#Azure api

class MinhaStreamListener(tw.StreamListener):

    def on_status(self, status):
        print(status.user.screen_name)
        print(status.text)
        print('-----')
        print('\n')

minhaStream = tw.Stream(auth = auth, listener=MinhaStreamListener())
minhaStream.filter(track=famosos)

#!pip install azure-cognitiveservices-vision-computervision

from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from msrest.authentication import CognitiveServicesCredentials
credenciais = CognitiveServicesCredentials("231d28866f4f4f8085291bd649922c23")
client = ComputerVisionClient("https://westcentralus.api.cognitive.microsoft.com",credenciais)
client.api_version

from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes

url = "http://pbs.twimg.com/media/ECx6hK-WwAAPzeE.jpg"

analize_de_imagem = client.analyze_image(url,VisualFeatureTypes.tags)

for tag in analize_de_imagem.tags:
  print(tag)

analise_celebridades = client.analyze_image_by_domain("celebrities", url, "en")

for celebridade in analise_celebridades.result["celebrities"]:
  print(celebridade['name'])
  print(celebridade['confidence'])

analise_celebridades.result["celebrities"]
descricao = client.describe_image(url,3,"en")
for caption in descricao.captions:
  print(caption.text)
  print(caption.confidence)

descricao.captions[0].text


class MinhaStreamListener(tw.StreamListener):

    def on_status(self, status):
        print("Usuário:", status.user.screen_name)
        print("Texto:", status.text)

        if 'media' in status.entities:
            url = status.entities['media'][0]['media_url']
            print("URL: ", url)

            analise_celebridades = client.analyze_image_by_domain("celebrities", url, "en")
            lista_celebridades = [celebridade['name'] for celebridade in analise_celebridades.result["celebrities"]]
            print(lista_celebridades)

            descricao = client.describe_image(url, 1, "en")
            texto_descricao = descricao.captions[0].text
            print("Descricao: ", texto_descricao)

        print('-----')
        print('\n')

minhaStream = None
minhaStream = tw.Stream(auth = auth, listener=MinhaStreamListener())
minhaStream.filter(follow=['917548352446791685'])
