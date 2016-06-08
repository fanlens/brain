#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from pykismet3 import Akismet

from config.db import Config

config = Config('akismet')

a = Akismet(blog_url="", user_agent="Testing/Testing")

a.api_key = config['api_key']

print(
a.check({'user_ip': '127.0.0.1',
         'user_agent': 'Mozilla/Firefox',
         'referrer': 'unknown',
         'comment_content': """Batman V Superman : Dawn of Justice
http://sendalgiceng21.blogspot.com/2016/03/batman-v-superman-dawn-of-justice.html

Captain America Civil War
http://sendalgiceng21.blogspot.com/2016/03/captain-america-civil-war.html

XXX : The Return Of Xander Cage
http://sendalgiceng21.blogspot.com/2016/03/xxx-return-of-xander-cage.html

Deadpool
http://sendalgiceng21.blogspot.com/2016/03/ddeadpool.html

Zoolander 2 Full Movie
http://sendalgiceng21.blogspot.com/2016/03/zoolsnder-2-full-movie.html

Pirattes Of The Caribean {2017}
http://sendalgiceng21.blogspot.com/2016/03/pirattes-of-caribean-2017.html

Barbershoop
http://sendalgiceng21.blogspot.com/2016/03/barbershoop.html

The Jungle Book
http://sendalgiceng21.blogspot.com/2016/03/the-jungle-book.html

Warcraft The Movie
http://sendalgiceng21.blogspot.com/2016/03/warcraft-movie.html

Creed
http://sendalgiceng21.blogspot.com/2016/03/creed.html

Criminal
http://sendalgiceng21.blogspot.com/2016/03/criminal.html

Daredevil
http://sendalgiceng21.blogspot.com/2016/03/daredevil.html

Dead 7
http://sendalgiceng21.blogspot.com/2016/03/dead-7.html

Fast 8 New Roads Ahead
http://sendalgiceng21.blogspot.com/2016/03/fast-8-new-roads-ahead.html

Gods Of Egypt
http://sendalgiceng21.blogspot.com/2016/03/gods-of-egypt.html

Guardian Of The Galaxy
http://sendalgiceng21.blogspot.com/2016/03/guardian-of-galaxy.html

Fifty Shades of Black
http://sendalgiceng21.blogspot.com/2016/03/fifty-shades-of-black.html

Jack Reacher
http://sendalgiceng21.blogspot.com/2016/03/jack-reacher.html

Mechanic
http://sendalgiceng21.blogspot.com/2016/03/mechanic.html

#100% free.. enjoy youre time.. just do it!!""",
         'comment_author': 'Wicky',
         'is_test': 1,
         })
)
