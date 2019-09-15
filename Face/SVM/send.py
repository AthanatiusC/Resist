import requests
import json
import time
import attendance

api = "http://172.10.0.52/api/v2"
# api = "http://192.168.8.8/api/v2"

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
                r = requests.post(api+"/attendance/",files=image,data=json,headers={"Accept":"application/json"},timeout=7)
                message = r.json()
                print(" [ SYSTEM ]: {}".format("Success!"),end="\n")
                attendance.sent.update({"sent":True})
        except (requests.ConnectTimeout,Exception):
                attendance.sent.update({"sent":True})
                for i in range(3,-1,-1):
                        print(" [ SYSTEM ]: Connection Timed Out! Retrying in {} Seconds".format(i),end="\r")
                        time.sleep(1)
                print("",end="\n")
                print(" [ SYSTEM ]: Retrying...")

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
                print(r.content)
                # y = json.loads(r.content)
                # if y["status"] == "true": 
                #         print("Deteksi Berhasil")
        except Exception as e:
                print(e)