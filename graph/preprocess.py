def add_comma_end_of_line(input_filename: str, output_filename: str):
	f = open(input_filename, "r")
	data = ""
	l = f.readline()
	while(l):
		l = l.strip()
		l += ",\n"
		data += l
		l = f.readline()
	o = open(output_filename, "w")
	o.writelines(data)
	o.close()
	f.close()

add_comma_end_of_line("./data/2024-01-09_raw.txt", "data/2024-01-09_locations.py")

def preprocess_(i: int):
	f = open(f"location{i}b_raw.py", "r")
	location = ""
	l = f.readline()
	l = l.strip()
	location = location + l + " = [\n"
	l = f.readline()
	while(l):
		t = l.strip()
		t += ",\n"
		location += t
		l = f.readline()
	f.close()
	a = open(f"./location{i}b.py", "w")
	location = location[: -2]
	location += "]"
	a.writelines(location)
	a.close()

# addComma(24)



'''
example of pre-processed data:
 ev location = 0.000000
 ev location = 0.000800
 ev location = 0.002400
 ev location = 0.004800
 ev location = 0.008000
 ev location = 0.012000
 ev location = 0.016800
 ev location = 0.022400
'''

def retriveSimpleControlData(source_file: str, target_file: str):
	f = open("./data/"+source_file, "r")
	location = "sl_full = [\n"
	l = f.readline()
	while(l):
		t = l[15:].strip()
		t += ",\n"
		location += t
		l = f.readline()
	f.close()
	location = location[: -2]
	location += "]"
	a = open("./data/"+target_file+".py", "w")
	a.writelines(location)
	a.close()


