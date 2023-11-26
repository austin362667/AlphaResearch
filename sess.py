import requests
import os, typing, time, sys
import pandas as pd
from urllib.parse import urljoin
from time import sleep
from getpass import getpass

from alpha_module import AlphaStage, Alpha, PNL_FOLDER, TVR_FOLDER

API_BASE = "https://api.worldquantbrain.com"

import pandas as pd

def get_alpha_pnl(s, alpha_id):
    while True:
        result = s.get(
            "https://api.worldquantbrain.com/alphas/" + alpha_id + "/recordsets/pnl"
        )
        if "retry-after" in result.headers:
            time.sleep(float(result.headers["Retry-After"]))
        else:
            break
    pnl = result.json().get("records", 0)
    if pnl == 0:
        return pd.DataFrame()
    pnl_df = (
        pd.DataFrame(pnl, columns=["Date", "Pnl"])
        .assign(
            alpha_id=alpha_id, Date=lambda x: pd.to_datetime(x.Date, format="%Y-%m-%d")
        )
        .set_index("Date")
    )
    return pnl_df["Pnl"]#.diff(1) # .tolist()


def get_alpha_tvr(s, alpha_id):
    while True:
        result = s.get(
            "https://api.worldquantbrain.com/alphas/" + alpha_id + "/recordsets/turnover"
        )
        if "retry-after" in result.headers:
            time.sleep(float(result.headers["Retry-After"]))
        else:
            break
    tvr = result.json().get("records", 0)
    if tvr == 0:
        return pd.DataFrame()
    tvr_df = (
        pd.DataFrame(tvr, columns=["Date", "Turnover"])
        .assign(
            alpha_id=alpha_id, Date=lambda x: pd.to_datetime(x.Date, format="%Y-%m-%d")
        )
        .set_index("Date")
    )
    return tvr_df["Turnover"] # .tolist()

class BrainSession:

    def __init__(self) -> None:
        lstatus = None
        cnt = 0
        stime = 0
        while lstatus is None or lstatus == requests.status_codes.codes.unauthorized or stime == 0:
            time.sleep(5)
            s: requests.Session = requests.Session()
            credential_email = os.environ.get('WQ_CREDENTIAL_EMAIL')
            credential_password = os.environ.get('WQ_CREDENTIAL_PASSWORD')
            s.auth = (credential_email, credential_password)
            r = s.post(f"{API_BASE}/authentication")

            lstatus = r.status_code
            stime = self.check_session_timeout()
            cnt += 1
            print(cnt, lstatus, stime)
            self.sess: requests.Session = s

    def check_session_timeout(self):
        try:
            result = self.sess.get(f"{API_BASE}/authentication").json()["token"]["expiry"]
            return float(result)
        except:
            return 0.0

    # def login(self) -> int:
    #     res: requests.Response = self.sess.post(f"{API_BASE}/authentication")
    #     if res.status_code == requests.status_codes.codes.unauthorized:
    #         if res.headers["WWW-Authenticate"] == "persona":
    #             print("Complete biometrics authentication: ", urljoin(res.url, res.headers["Location"]))
    #             input("Press any key To Continue..")
    #             self.sess.post(urljoin(res.url, res.headers["Location"]))
    #             print("Authentication success.")
    #         else:
    #             print("Login fail: incorrect email or password.")
    #             return 401
    #     return 

    def stream_simulation(self, max_sim = 1):
        while True:
            while len(os.listdir(AlphaStage.PENDING.value)) > 0 and len(os.listdir(AlphaStage.RUNNING.value)) < max_sim:
                pending_files = [os.path.join(AlphaStage.PENDING.value, file) for file in os.listdir(AlphaStage.PENDING.value)]
                for pending_file in pending_files:
                    pending_alpha = Alpha.load_from_disk(file_path=pending_file)
                    post_res: requests.Response = self.sess.post(f"{API_BASE}/simulations", json=pending_alpha.payload)
                    time.sleep(0.1)
                    if post_res.status_code == 201:
                        location_url = post_res.headers["Location"]

                        pending_alpha.update_stage(alpha_stage=AlphaStage.RUNNING)
                        pending_alpha.update_running_data(running_data=location_url)
                    else:
                        print("Post simulation error: ")
                        print(post_res.json())
                        sys.exit(0)
                    pending_alpha.save_to_disk()
                    os.remove(pending_file)
                    break
            if len(os.listdir(AlphaStage.RUNNING.value)) > 0:
                for running_file in [os.path.join(AlphaStage.RUNNING.value, file) for file in os.listdir(AlphaStage.RUNNING.value)]:
                    running_alpha: Alpha = Alpha.load_from_disk(file_path=running_file)
                    sim_res: requests.Response = self.sess.get(running_alpha.running_data)
                    if 'Retry-After' in sim_res.headers:
                        wait_time = float(sim_res.headers['Retry-After'])
                    else:
                        wait_time = 0
                    time.sleep(wait_time)
                    if wait_time <= 0:
                        sim_res_json = sim_res.json()
                        alpha_id = sim_res_json.get("alpha", 0)
                        if alpha_id != 0:
                            out_res: requests.Response = self.sess.get(f"{API_BASE}/alphas/{alpha_id}")
                            out_res_json = out_res.json()
                            running_alpha.update_response_data(response_data=out_res_json)
                            running_alpha.update_stage(alpha_stage=AlphaStage.COMPLETED)

                            pnl_path = os.path.join(PNL_FOLDER, running_alpha.name)
                            pnl_df = get_alpha_pnl(self.sess, running_alpha.response_data['id'])
                            pnl_df.to_csv(pnl_path)
                            running_alpha.update_pnl_path(pnl_path=pnl_path)

                            tvr_path = os.path.join(TVR_FOLDER, running_alpha.name)
                            tvr_df = get_alpha_tvr(self.sess, running_alpha.response_data['id'])
                            tvr_df.to_csv(tvr_path)
                            running_alpha.update_tvr_path(tvr_path=tvr_path)

                        running_alpha.save_to_disk()
                        os.remove(running_file)
            time.sleep(10)
while True:
    worker = BrainSession()
    worker.stream_simulation(max_sim=10)
    time.sleep(30)
