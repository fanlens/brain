#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pycld2
from lib.alchemy.alchemyapi import AlchemyAPI

from config.db import Config

from db import get_session, Session


class AlchemyApiMem(AlchemyAPI):
    def __init__(self, key: str):
        assert len(key) == 40
        self.apikey = key


config = Config("alchemyapi")

# post_with_comments = {}
# with get_session() as session:
#     for post_id, comments in session.execute("""
# select post_id, array_agg(comment::jsonb->>'message')
# from facebook_comments
# where page = 'redbull' and char_length(comment::jsonb->>'message') >= 140
# group by post_id
# having count(*) > 4
#     """):
#         post_with_comments[post_id] = [comment for comment in comments if pycld2.detect(comment)[2][0][1] == 'en']


# try to build up a word cloud
alchemyapi = AlchemyApiMem(config['api_key'])
response = alchemyapi.combined('text', """The tax funding is actually an anti corruption measure. By using tax money, and limiting funds allowed from people and companies, it helps stop companies from funding a party to the point that they control the party.
So right now you are probably thinking "But isn't that exactly what super-pacs are doing", and the answer is yes, but to a much lesser degree. Super-pacs are not allowed to give directly to a campaign due to the limits mentioned above. We do have the right to free speech though, so there is nothing wrong with using money to promote the person who you like as long as you don't give more than the limit directly to them.""", options={'sentiment': 1})  # type: dict
#image = 'https://pbs.twimg.com/media/Cheuhy5W0AAypE9.jpg:large'
#response = alchemyapi.imageTagging('url', image)  # type: dict

for key, value in response.items():
    print(key + ':')
    if isinstance(value, list):
        for v in value:
            print('\t', v)
    else:
        print('\t', value)
