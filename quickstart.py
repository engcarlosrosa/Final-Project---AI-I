from datetime import datetime
import os

#import threading
#from threading import Thread

import pyttsx


from apiclient.discovery import build
from httplib2 import Http
import oauth2client
from oauth2client import client
from oauth2client import tools
from Tkinter import *

from multiprocessing import Process


root = Tk()

text1 = Text(root, height=200, width=60)
photo=PhotoImage(file='./IMG_4330.PNG')
text1.insert(END,'\n')
text1.image_create(END, image=photo)

text1.pack(side=LEFT)

text2 = Text(root, height=200, width=500)
scroll = Scrollbar(root, command=text2.yview)
text2.configure(yscrollcommand=scroll.set)
text2.tag_configure('bold_italics', font=('Arial', 12, 'bold', 'italic'))
text2.tag_configure('big', font=('Verdana', 20, 'bold'))
text2.tag_configure('color', foreground='#476042', 
                        font=('Tempus Sans ITC', 12, 'bold'))
text2.tag_bind('follow', '<1>', lambda e, t=text2: t.insert(END, "Not now, maybe later!"))
text2.insert(END,'\nCarlos,\n', 'big')
text2.insert(END,'Seus 10 proximos eventos sao:\n', 'big')
engine = pyttsx.init()
rate = engine.getProperty('rate')
engine.setProperty('rate', rate-10000)
volume = engine.getProperty('volume')
engine.setProperty('volume', volume+10)
pyttsx.voice.Voice.languages = ['Portuguese']

try:
    import argparse
    flags = argparse.ArgumentParser(parents=[tools.argparser]).parse_args()
except ImportError:
    flags = None

SCOPES = 'https://www.googleapis.com/auth/calendar.readonly'
CLIENT_SECRET_FILE = 'client_secret.json'
APPLICATION_NAME = 'Calendar API Quickstart'


def get_credentials():
    """Gets valid user credentials from storage.

    If nothing has been stored, or if the stored credentials are invalid,
    the OAuth2 flow is completed to obtain the new credentials.

    Returns:
        Credentials, the obtained credential.
    """
    home_dir = os.path.expanduser('~')
    credential_dir = os.path.join(home_dir, '.credentials')
    if not os.path.exists(credential_dir):
        os.makedirs(credential_dir)

    credential_path = os.path.join(credential_dir,
                                   'calendar-api-quickstart.json')
    store = oauth2client.file.Storage(credential_path)
    credentials = store.get()
    if not credentials or credentials.invalid:
        flow = client.flow_from_clientsecrets(CLIENT_SECRET_FILE, SCOPES)
        flow.user_agent = APPLICATION_NAME
        if flags:
            credentials = tools.run_flow(flow, store, flags)
        else: # Needed only for compatability with Python 2.6
            credentials = tools.run(flow, store)
        print 'Storing credentials to ' + credential_path
    return credentials


def main():
    """Shows basic usage of the Google Calendar API.

    Creates a Google Calendar API service object and outputs a list of the next
    10 events on the user's calendar.
    """
    credentials = get_credentials()
    service = build('calendar', 'v3', http=credentials.authorize(Http()))

    now = datetime.utcnow().isoformat() + 'Z' # 'Z' indicates UTC time
    print 'Getting the upcoming 10 events'
    eventsResult = service.events().list(
        calendarId='primary', timeMin=now, maxResults=10, singleEvents=True,
        orderBy='startTime').execute()
    events = eventsResult.get('items', [])

    if not events:
        print 'No upcoming events found.'
    for event in events:
        start = event['start'].get('dateTime', event['start'].get('date'))
        print start, event['summary']
        engine.say(start)
        engine.say(event['summary'])
        
        text2.insert(END, start, 'color')
        text2.insert(END, '\n', 'follow')
        text2.insert(END, event['summary'], 'color')
        text2.insert(END, '\n\n\n', 'follow')
        text2.pack(side=LEFT)
        scroll.pack(side=RIGHT, fill=Y)
    '''    
    if __name__ == '__main__':
      p1 = Process(target=root.mainloop())
      p1.start()
      p2 = Process(target=engine.runAndWait())
      p2.start()
      p1.join()
      p2.join()
    '''
    def runInParallel(*fns):
          proc = []
          for fn in fns:
            p = Process(target=fn)
            p.start()
            #proc.append(p)
          #for p in proc:
            #p.join()

    runInParallel(root.mainloop)#, engine.runAndWait)  
    #if __name__ == '__main__':
     #   Thread(target = root.mainloop()).start()
     #   Thread(target = engine.runAndWait()).start()
    #print dir(engine)
    
    #root.mainloop()        
    engine.runAndWait()
     

if __name__ == '__main__':
    main()

