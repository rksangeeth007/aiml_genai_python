import re

format_pat= re.compile(
r"(?P<host>[\d\.]+)\s"
r"(?P<identity>\S*)\s"
r"(?P<user>\S*)\s"
r"\[(?P<time>.*?)\]\s"
r'"(?P<request>.*?)"\s'
r"(?P<status>\d+)\s"
r"(?P<bytes>\S*)\s"
r'"(?P<referer>.*?)"\s'
r'"(?P<user_agent>.*?)"\s*'
)

logPath = "/Users/s0h0902/BigDataFinal/Repos/ML_GenAI_Python_Udemy/src/main/resources/access_log.txt"

# URLCounts = {}
#
# with open(logPath, "r") as f:
#     for line in (l.rstrip() for l in f):
#         match= format_pat.match(line)
#         if match:
#             access = match.groupdict()
#             request = access['request']
#             (action, URL, protocol) = request.split()
#             if URL in URLCounts:
#                 URLCounts[URL] = URLCounts[URL] + 1
#             else:
#                 URLCounts[URL] = 1


# results = sorted(URLCounts, key=lambda i: int(URLCounts[i]), reverse=True)
#
# for result in results[:20]:
#     print(result + ": " + str(URLCounts[result]))


# URLCounts = {}
#
# with open(logPath, "r") as f:
#     for line in (l.rstrip() for l in f):
#         match= format_pat.match(line)
#         if match:
#             access = match.groupdict()
#             request = access['request']
#             fields = request.split()
#             if (len(fields) != 3):
#                 print(fields)

# URLCounts = {}
#
# with open(logPath, "r") as f:
#     for line in (l.rstrip() for l in f):
#         match= format_pat.match(line)
#         if match:
#             access = match.groupdict()
#             request = access['request']
#             fields = request.split()
#             if (len(fields) == 3):     #We add this condition to remove those requests which doesn’t have 3 fields.
#                 URL = fields[1]
#                 if URL in URLCounts:
#                     URLCounts[URL] = URLCounts[URL] + 1
#             else:
#                 URLCounts[URL] = 1
#
# results = sorted(URLCounts, key=lambda i: int(URLCounts[i]), reverse=True)
#
# for result in results[:20]:
#     print(result + ": " + str(URLCounts[result]))


# URLCounts = {}
#
# with open(logPath, "r") as f:
#     for line in (l.rstrip() for l in f):
#         match= format_pat.match(line)
#         if match:
#             access = match.groupdict()
#             request = access['request']
#             fields = request.split()
#             if (len(fields) == 3):
#                 (action, URL, protocol) = fields
#                 if (action == 'GET'):
#                     if URL in URLCounts:
#                         URLCounts[URL] = URLCounts[URL] + 1
#                 else:
#                     URLCounts[URL] = 1
#
# results = sorted(URLCounts, key=lambda i: int(URLCounts[i]), reverse=True)
#
# for result in results[:20]:
#     print(result + ": " + str(URLCounts[result]))

# UserAgents = {}
#
# with open(logPath, "r") as f:
#     for line in (l.rstrip() for l in f):
#         match= format_pat.match(line)
#         if match:
#             access = match.groupdict()
#             agent = access['user_agent']
#             if agent in UserAgents:
#                 UserAgents[agent] = UserAgents[agent] + 1
#             else:
#                 UserAgents[agent] = 1
#
# results = sorted(UserAgents, key=lambda i: int(UserAgents[i]), reverse=True)
#
# for result in results:
#     print(result + ": " + str(UserAgents[result]))

# URLCounts = {}
#
# with open(logPath, "r") as f:
#     for line in (l.rstrip() for l in f):
#         match= format_pat.match(line)
#         if match:
#             access = match.groupdict()
#             agent = access['user_agent']
#             if (not('bot' in agent or 'spider' in agent or
#             'Bot' in agent or 'Spider' in agent or
#             'W3 Total Cache' in agent or agent =='-')):
#                 request = access['request']
#                 fields = request.split()
#                 if (len(fields) == 3):
#                     (action, URL, protocol) = fields
#                     if (action == 'GET'):
#                         if URL in URLCounts:
#                             URLCounts[URL] = URLCounts[URL] + 1
#                         else:
#                             URLCounts[URL] = 1
#
# results = sorted(URLCounts, key=lambda i: int(URLCounts[i]), reverse=True)
#
# for result in results[:20]:
#     print(result + ": " + str(URLCounts[result]))

URLCounts = {}

with open(logPath, "r") as f:
    for line in (l.rstrip() for l in f):
        match= format_pat.match(line)
        if match:
            access = match.groupdict()
            agent = access['user_agent']
            if (not('bot' in agent or 'spider' in agent or
                    'Bot' in agent or 'Spider' in agent or
                    'W3 Total Cache' in agent or agent =='-')):
                request = access['request']
                fields = request.split()
                if (len(fields) == 3):
                    (action, URL, protocol) = fields
                    if (URL.endswith("/") or URL not in '/feed'):     #Added to remove /feed from the data.
                        if (action == 'GET'):
                            if URL in URLCounts:
                                URLCounts[URL] = URLCounts[URL] + 1
                            else:
                                URLCounts[URL] = 1

results = sorted(URLCounts, key=lambda i: int(URLCounts[i]), reverse=True)

for result in results[:20]:
    print(result + ": " + str(URLCounts[result]))
