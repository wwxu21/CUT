import json
import random
import os
random.seed(42)

def process_open_summary(addr, addr_save):
    f = []
    count = 0 
    with open(addr) as reader:
        for line in reader:
            x = json.loads(line)
            x = x['data']
            x_passage = x['passage']['text']
            
            for question in x['questions']:
                x_question = question['question']
                
                x_instruction = f'{x_passage}\n{x_question}'
                rankings = question['rankings']
                max_index = rankings.index(max(rankings))
                ans = question['answers'][max_index]
                x_out = ans['answer']
                if ('skip' in  ans['feedback'] and ans['feedback']['skip'] == True) or x_out == "":
                    continue
                
                cirtique = ans['feedback']['critiques']
                count += 1
                if cirtique is None or cirtique == []:
                    assert ans['feedback']['rating'] == 7
                    x_judgment = None
                    x_i_ans = None
                    x_score =  None
                    new_x = {
                                'output':x_out,
                                'input':None,
                                'instruction':x_instruction,
                                'reason':None,
                                'judgment': x_judgment,
                                'i_ans':x_i_ans,
                                'score':x_score,
                            }
                    
                    f.append(new_x)
                else:
                    # cirtique = cirtique[0]
                    x_judgment = ""
                    for critique_one in cirtique:
                        x_judgment = x_judgment + " " + critique_one['text']
                        x_i_ans = critique_one['refinement']
                    if x_i_ans == "":
                        continue
                    x_score = 1
                    
                    x_judgment = x_judgment.strip()
                    new_x = {
                            'output':x_out,
                            'input':None,
                            'instruction':x_instruction,
                            'reason':None,
                            'judgment': x_judgment,
                            'i_ans':x_i_ans,
                            'score':x_score,
                        }
                    f.append(new_x)
    print(count)
    with open(addr_save, 'w') as writer:
        json.dump(f, writer, ensure_ascii=False, indent=2)
    return f

def process_Shepherd(addr, feedback_addr, addr_save):
    f = []
    data = []
    feedback = []
    with open(addr) as reader:
        for line in reader:
            x = json.loads(line)
            data.append(x)
    with open(feedback_addr) as reader:
        for line in reader:
            x = json.loads(line)
            feedback.append(x)

    for i in range(len(data)):
        data_one = data[i]
        feedback_one = feedback[i]
        x_out = feedback_one['answer']
        x_instruction = feedback_one['question']
        x_judgment = feedback_one['feedback']
        x_socre = 1
        x_i_ans = data_one['metadata']['output_correct']
        assert data_one['metadata']['context'] == feedback_one['question']
        assert data_one['metadata']['output_candidate'] == feedback_one['answer']
        x_ref = None
        new_x = {
                    'output':x_out,
                    'input':None,
                    'instruction':x_instruction,
                    'reason':None,
                    'judgment': x_judgment,
                    'i_ans':x_i_ans,
                    'score':x_socre,
                    "ref":x_ref,
            }
        f.append(new_x)
    with open(addr_save, 'w') as writer:
        json.dump(f, writer, ensure_ascii=False, indent=2)
    return f

def combine_ians(addr1, addr2):
    with open(addr1, ) as reader:
        f1 = json.load(reader)
    with open(addr2, ) as reader:
        f2 = json.load(reader)
    new_f = []
    for i in range(len(f1)):
        new_one = f2[i]
        new_one['i_ans'] = f1[i]['i_ans']
        new_f.append(new_one)
    with open(addr2, 'w') as writer:
        json.dump(new_f, writer, ensure_ascii=False, indent=2)
    return new_f

if __name__ == "__main__":
    data_file = "data/open_summary/critiques.train.jsonl"
    save_file = "data/open_summary/train-alpaca.json"
    if not os.path.exists("data/open_summary"):
        os.mkdir("data/open_summary")
    process_open_summary(data_file, save_file)
    data_file = "data/open_summary/critiques.test.jsonl"
    save_file = "data/open_summary/test-alpaca.json"
    process_open_summary(data_file, save_file)
    data_file = "data/Shepherd/human_data_raw.jsonl"
    feedback_file = "data/Shepherd/human_data_for_model.jsonl"
    save_file = "data/Shepherd/train-alpaca.json"
    if not os.path.exists("data/Shepherd"):
        os.mkdir("data/Shepherd")
    f = process_Shepherd(data_file, feedback_file, save_file)