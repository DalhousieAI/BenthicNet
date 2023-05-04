import json
from collections import defaultdict

import pandas as pd
import requests


class InvalidAphiaIDException(Exception):
    def __init__(self, i):
        self.i = i

    def __str__(self):
        return (
            "The server did not respond to the request for given Aphia ID : {}!".format(
                self.i
            )
        )


def aphiaID2taxonomy(identity):
    """Fetch classification information in a hierarchical string form using Aphia IDs.

    Parameters
    ----------
    identity : int or str
        The Aphia ID of WoRMS taxon.

    Returns
    -------
    res : str
        Consists of the hierarchical classification going from higher to lower level taxonomies.

    Raises
    ------
    InvalidAphiaIDException
        If the response status is other than 200 (which means the response was not successful),
        we raise an exception saying the server did not respond to the request.

    References
    ----------
    .. [1] "WoRMS RESTful service", https://www.marinespecies.org/rest/AphiaClassificationByAphiaID
    """
    identity = str(identity)
    api_url = (
        "https://www.marinespecies.org/rest/AphiaClassificationByAphiaID/" + identity
    )
    response = requests.get(api_url)

    # If the API request was successful, extract the URL data from the response
    if response.status_code == 200:
        # Collect json data from response obtained through the api and convert it to defaultdict
        data = json.loads(response.content)
        d = defaultdict(dict, data)

        # res will contain the resultant string
        res = ""

        # Iterating till there are no children in the classifications
        while True:
            # Extracting scientific name from response
            res += "{} > ".format(str(d["scientificname"]))
            if d["child"] is not None:
                d = d["child"]
            else:
                break

        res = res[:-3]
        return res

    # If the API request failed, raise InvalidAphiaIDException
    else:
        raise InvalidAphiaIDException(identity)


def taxon_status(identity):
    """Show whether a taxon associated with a given Aphia ID is accepted or not.

    Parameters
    ----------
    identity : int or str
        The Aphia ID of WoRMS taxon.

    Returns
    -------
    bool
        A boolean value
          False indicates that the taxon is unaccepted.
          True indicates that the taxon is accepted.

    Raises
    ------
    InvalidAphiaIDException
        If the response status is other than 200 (which means the response was not successful),
        we raise an exception saying the server did not respond to the request

    References
    ----------
    .. [1] "WoRMS RESTful service", https://www.marinespecies.org/rest/AphiaRecordByExternalID
    """
    identity = str(identity)
    api_url = "https://www.marinespecies.org/rest/AphiaRecordByAphiaID/" + identity
    response = requests.get(api_url)

    # If the API request was successful, extract the URL data from the response
    if response.status_code == 200:
        # Collect json data from response obtained through the api and convert it to defaultdict
        data = json.loads(response.content)
        d = defaultdict(dict, data)

        if d["status"] == "accepted":
            return True
        return False

    # If the API request failed, raise InvalidAphiaIDException
    else:
        raise InvalidAphiaIDException(identity)


def aphiaID2mapping(identity):
    """Associate every taxonomy to its level for a given Aphia ID.

    Parameters
    ----------
    identity : int or str
        The Aphia ID of WoRMS taxon.

    Returns
    -------
    res : defaultdict
        Consists of different classifications as keys and corresponding levels as values.

    Raises
    ------
    InvalidAphiaIDException
        If the response status is not 200 (the response was not successful),
        we raise an exception saying the server did not respond to the request.

    References
    ----------
    .. [1] "WoRMS RESTful service", https://www.marinespecies.org/rest/AphiaClassificationByAphiaID

    Example
    -------
    Input : 140892
    Output : {'Biota': 'Superdomain',
             'Animalia': 'Kingdom',
             'Mollusca': 'Phylum',
             'Scaphopoda': 'Class',
             'Gadilida': 'Order',
             'Gadilimorpha': 'Suborder',
             'Pulsellidae': 'Family',
             'Pulsellum': 'Genus',
             'Pulsellum affine': 'Species'}
    """
    identity = str(identity)
    api_url = (
        "https://www.marinespecies.org/rest/AphiaClassificationByAphiaID/" + identity
    )
    response = requests.get(api_url)

    # If the API request was successful, extract the URL data from the response
    if response.status_code == 200:
        # Collect json data from response obtained through the api and convert it to defaultdict
        data = json.loads(response.content)
        d = defaultdict(dict, data)

        res = defaultdict(str)

        # Iterating till there are no children in the classifications
        while True:
            # Assigning levels to corresponding classifications
            res[d["scientificname"]] = d["rank"]

            if d["child"] is not None:
                d = d["child"]
            else:
                break
        return res

    # If the API request failed, raise InvalidAphiaIDException
    else:
        raise InvalidAphiaIDException(identity)


def aphiaID2counts(ids):
    """Return a dictionary containing the count of each classification in a list of Aphia IDs.

    Parameters
    ----------
    ids : pandas Series
        Contains Aphia IDs from WoRMS.

    Returns
    -------
    resdict : defaultdict
        Contains counts of different classifications for a given list of Aphia IDs.

    Raises
    ------
    InvalidAphiaIDException
        If the response status is not 200 (the response was not successful),
        we raise an exception saying the server did not respond to the request.

    References
    ----------
    .. [1] "WoRMS RESTful service", https://www.marinespecies.org/rest/AphiaClassificationByAphiaID

    Example
    -------
    Input : pd.Series([140892,124528,382936])
    Output : {'Biota': 3,
             'Animalia': 3,
             'Mollusca': 1,
             'Scaphopoda': 1,
             'Gadilida': 1,
             'Gadilimorpha': 1,
             'Pulsellidae': 1,
             'Pulsellum': 1,
             'Pulsellum affine': 1,
             'Echinodermata': 1,
             'Echinozoa': 1,
             'Holothuroidea': 1,
             'Actinopoda': 1,
             'Holothuriida': 1,
             'Holothuriidae': 1,
             'Holothuria': 1,
             'Holothuria (Platyperona)': 1,
             'Holothuria (Platyperona) sanctori': 1,
             'Arthropoda': 1,
             'Crustacea': 1,
             'Multicrustacea': 1,
             'Malacostraca': 1,
             'Eumalacostraca': 1,
             'Eucarida': 1,
             'Decapoda': 1,
             'Pleocyemata': 1,
             'Achelata': 1,
             'Palinuroidea': 1,
             'Scyllaridae': 1,
             'Scyllarinae': 1,
             'Bathyarctus': 1,
             'Bathyarctus formosanus': 1}
    """
    resdict = defaultdict(int)
    for item in ids.iteritems():
        identity = str(item)
        api_url = (
            "https://www.marinespecies.org/rest/AphiaClassificationByAphiaID/"
            + identity
        )
        response = requests.get(api_url)

        # If the API request was successful, extract the URL data from the response
        if response.status_code == 200:
            # Collect json data from response obtained through the api and convert it to defaultdict
            data = json.loads(response.content)
            d = defaultdict(dict, data)

            # Iterating till there are no children in the classifications
            while True:
                # Incrementing count of classification
                resdict[d["scientificname"]] += 1
                if d["child"] is not None:
                    d = d["child"]
                else:
                    break

        # If the API request failed, raise InvalidAphiaIDException
        else:
            raise InvalidAphiaIDException(identity)

    return resdict
