import requests
import json
import time

# api = "http://103.247.219.34/api/v2"
api = "http://172.10.0.52:8000/api/v2"

def get_identity(uid):
        try:
                r = requests.get(api+"/user/"+uid)
                y = json.loads(r.content)
                data = {"status":y["status"],"name":y["name"]}
                x = json.dumps(data)
                return x
        except Exception as e:
                print(e)

def post_attendance(json,image):
    try:
        r = requests.post(api+"/attendance/",files=image,data=json,headers={"Accept":"application/json"})
        print(r.content)
    except Exception as e:
        print(e)

def post_register(image,json):
        try:
                r = requests.post(api+"/register/verify",files=image,data=json,headers={"Accept":"application/json"})
                print(r.content)
        except Exception as e:
                print(e)
def track(jsons):
        try:
                time.sleep(3)
                r = requests.post(api+"/detect/",data=jsons,headers={"Accept":"application/json"})
                print(r.code)
                # y = json.loads(r.content)
                # if y["status"] == "true": 
                #         print("Deteksi Berhasil")
        except Exception as e:
                print(e)