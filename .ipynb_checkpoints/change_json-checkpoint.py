import json

json_list = []
with open('result.json') as json_file:
    data = json.load(json_file)
    json_list.append(data)

#Now, output the list of json data into a single jsonl file
with open('output.jsonl', 'w') as outfile:
    for entry in json_list:
        json.dump(entry, outfile,ensure_ascii=False)
        outfile.write('\n')


