import pandas as pd
import requests


def aphiaID2taxonomy(identity):
    identity = str(identity)
    api_url = (
        "https://www.marinespecies.org/rest/AphiaClassificationByAphiaID/" + identity
    )
    response = requests.get(api_url)

    if response.status_code == 200:
        # If the API request was successful, extract the URL from the response
        d = dict(response.json())
        res = ""
        while True:
            res += str(d["scientificname"]) + " > "
            if d["child"] is not None:
                d = d["child"]
            else:
                break
        res = res[:-3]
        return res

    else:
        return "Failed to get a response"


def aphiaID2status(identity):
    # https://www.marinespecies.org/rest/AphiaRecordByExternalID

    identity = str(identity)
    api_url = "https://www.marinespecies.org/rest/AphiaRecordByAphiaID/" + identity
    response = requests.get(api_url)

    if response.status_code == 200:
        # If the API request was successful, extract the URL from the response
        d = dict(response.json())
        if d["status"] == "accepted":
            return True
        return False

    else:
        return "Failed to get a response"


def aphiaID2mapping(identity):
    identity = str(identity)
    api_url = (
        "https://www.marinespecies.org/rest/AphiaClassificationByAphiaID/" + identity
    )
    response = requests.get(api_url)

    if response.status_code == 200:
        # If the API request was successful, extract the URL from the response
        d = dict(response.json())
        res = {}
        while True:
            res[d["scientificname"]] = d["rank"]
            if d["child"] is not None:
                d = d["child"]
            else:
                break
        return res

    else:
        return "Failed to get a response"


def aphiaID2counts(ids):
    ids = ids.tolist()
    resdict = {}
    for identity in ids:
        identity = str(identity)
        api_url = (
            "https://www.marinespecies.org/rest/AphiaClassificationByAphiaID/"
            + identity
        )
        response = requests.get(api_url)

        d = dict(response.json())
        while True:
            if d["scientificname"] not in resdict.keys():
                resdict[d["scientificname"]] = 1
            else:
                resdict[d["scientificname"]] += 1

            if d["child"] is not None:
                d = d["child"]
            else:
                break

    return resdict
