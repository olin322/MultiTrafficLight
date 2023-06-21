

def getClassName(s: str) -> str:
	if ((s[0] == '<') & (s[-1] == '>')):
		s = s[1: -1]
	else:
		return None
	if (s[0:5] == "class"):
		s = s[6: ]
	else:
		return None
	if ((s[0] == "\'") & (s[-1] == "\'")):
		s = s[1: -1]
	else:
		return None
	if ('.' in s):
		return s.split('.')[1]
	return None