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