def addComma():
	f = open("locations.py", "r")
	location = ""
	l = f.readline()
	while(l):
		t = l + ","
		location += t
		l = f.readline()
	f.close()
	a = open("./updated_locations.py", "w")
	a.writelines(location)
	a.close()

addComma()