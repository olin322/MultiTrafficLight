def addComma():
	f = open("location13b.py", "r")
	location = ""
	l = f.readline()
	while(l):
		t = l.strip()
		t += ",\n"
		location += t
		l = f.readline()
	f.close()
	a = open("./updated_locations.py", "w")
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