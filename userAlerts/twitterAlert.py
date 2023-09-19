import requests
import tweepy
import logging

class TwitterCommunicator:
    def __init__(self):
        logging.debug("Initializing TwitterCommunicator")
        # Handle authorization for the Twitter API
        auth = tweepy.OAuthHandler("", 
                                    "")
        auth.set_access_token("",
                                "")
        # Add an instance of the api to class variables as well as the current authenticated 
        # user's id
        self.api = tweepy.API(auth)
        self.user = self.api.me().id

    '''
    Sends a dm to the user.
    @param {message}  REQUIRED     - The desired message to send
    @param {user}  NOT REQUIRED    - The id of the user to be dm'd. defaults to authenticated user 
    '''
    def directMessage(self, message, user=None):
        logging.debug("Sending direct message from TwitterCommunicator")
        if (user is not None):
            self.api.send_direct_message(user, message)
        
        else:
            self.api.send_direct_message(self.user, message)
