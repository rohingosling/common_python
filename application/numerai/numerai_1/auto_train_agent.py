import requests

url      = 'https://numer.ai/signin'
username = 'rohingosling@gmail.com'
password = 'Numerai?7ehwdlt'

def login ( url, username, password ):
    
    payload = { 'signin-email': username, 'signin-password': password }
    
    request = requests.post ( url, data = payload )
    
    print ( request.content )
    
def main ():
    print ( 'Hello World!' )

main()