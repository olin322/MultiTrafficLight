def addComma():
	i = 14
	f = open(f"location{i}b_raw.py", "r")
	location = ""
	l = f.readline()
	while(l):
		t = l.strip()
		t += ",\n"
		location += t
		l = f.readline()
	f.close()
	a = open(f"./location{i}b.py", "w")
	a.writelines(location)
	a.close()

addComma()

def trimPrefix():
	f = open("raw_data.py", "r")
	location = ""
	l = f.readline()
	while(l):
		t = l[15:].strip()
		t += ",\n"
		location += t
		l = f.readline()
	f.close()
	a = open("./new_simple_location.py", "w")
	a.writelines(location)
	a.close()

# trimPrefix()