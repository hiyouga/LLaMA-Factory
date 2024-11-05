"""
Reference Docs
[1] https://python.langchain.com/docs/integrations/document_loaders/pandas_dataframe
[2] https://js.langchain.com/docs/modules/data_connection/document_transformers/
[3] https://docs.pinecone.io/docs/overview
[4] https://python.langchain.com/docs/integrations/vectorstores/faiss
[5] https://github.com/openai/openai-cookbook/blob/main/examples/utils/embeddings_utils.py
[6] https://cookbook.openai.com/examples/batch_processing
[7] https://platform.openai.com/docs/guides/batch/getting-started
"""
# %%
import pandas as pd
import numpy as np
import re, os, json, time
import wandb
from tqdm import tqdm
import os.path as osp

### Load Data and Make Loader ##embeddings as unique 본문, 질문, paragraphs
def load_solvook_data(args):

    # load vector db
    with open(args.vector_db_path, "r") as f:
        data_dict = json.load(f)

    # load query
    query_db = pd.read_csv(args.query_path)
    for i in range(len(query_db)):
        query_db.loc[i, 'query'] = f'"본문" : "{query_db.loc[i, "본문"]}", "질문" : "{query_db.loc[i, "질문"]}"'
    
    if args.task == 2:
        query_db = query_db[query_db['relation']!=0].reset_index()


    return data_dict, query_db




def extract_label(text):
    info = dict()
    try:
        try:
            paragraph = re.search(r'<Paragraph>(.*?)<Paragraph>', text).group(1).strip()
        except:
            try:
                paragraph = re.search(r'<Paragraph>(.*?)</Paragraph>', text).group(1).strip()
            except:
                paragraph = re.search(r'<Paragraph>(.*?)<\Paragraph>', text).group(1).strip()
        info['paragraph'] = paragraph
    except:
        print("Failed to extract the paragraph label")
        info['paragraph'] = -9999
        
    try:
        try:
            skill = re.search(r'<Skill>(.*?)<Skill>', text).group(1).strip()
        except:
            try:
                skill = re.search(r'<Skill>(.*?)</Skill>', text).group(1).strip()
            except:
                skill = re.search(r'<Skill>(.*?)<\Skill>', text).group(1).strip()
        info['skill'] = skill
    except:
        print("Failed to extract the skill label")
        info['skill'] = -9999
        
    try:
        try:
            method = re.search(r'<Method>(.*?)<Method>', text).group(1).strip()
        except:
            try:
                method = re.search(r'<Method>(.*?)</Method>', text).group(1).strip()
            except:
                method = re.search(r'<Method>(.*?)<\Method>', text).group(1).strip()
        info['method'] = method
    except:
        print("Failed to extract the method label")
        info['method'] = -9999
        
    try:
        try:
            relation = re.search(r'<Relation>(.*?)<Relation>', text).group(1).strip()
        except:
            try:
                relation = re.search(r'<Relation>(.*?)</Relation>', text).group(1).strip()
            except:
                relation = re.search(r'<Relation>(.*?)<\Relation>', text).group(1).strip()
        info['relation'] = relation
    except:
        print("Failed to extract the relation label")
        info['relation'] = -9999
        
    try:
        try:
            description = re.search(r'<Description>(.*?)<Description>', text).group(1).strip()
        except:
            try:
                description = re.search(r'<Description>(.*?)</Description>', text).group(1).strip()
            except:
                description = re.search(r'<Description>(.*?)<\Description>', text).group(1).strip()
        info['description'] = description
    except:
        print("Failed to extract the description label")
        info['description'] = -9999
    
    return info
    

    
    
## generation
def generation(args, retriever_dict, query_db):

    from openai import OpenAI
    client = OpenAI(api_key=args.openai_api_key)
    
    top_content_list = list()
    tasks = list()
    print("Start making prompts with top-k contents to send them into batch query")
    idx_list = list()
    id_list = list(); textbook_list = list(); unit_list = list(); story_list = list(); paragraph_list = list()
    skill_list = list(); method_list = list(); relation_list = list(); description_list = list()
    answer_list = list()
    for idx in tqdm(range(len(query_db))):
        #############################################################################
        ## Top-K search
        #############################################################################
        if args.task in [1, 2]:
            top_mt = retriever_dict['mt_db_retriever'].invoke(query_db['본문'][idx])          # 본문 v.s. 본문*
            top_parap = retriever_dict['parap_db_retriever'].invoke(query_db['본문'][idx])    # paragraph v.s. 본문*
        top_ques = retriever_dict['ques_db_retriever'].invoke(query_db['질문'][idx])      # 질문 v.s. 질문*
        
        top = top_ques
        if args.task in [1, 2]:
            top += top_mt + top_parap
        
        ## Get pair
        top_content = list()
        for k in range(len(top)):
            top_content_ = f"["
            if args.task in [1, 2]:
                # 본문 id, 본문 (paragraphs)
                top_content_ = f"'본문 id': '{top[k].metadata['textbook_id']}_{top[k].metadata['unit_id']}_{top[k].metadata['story_id']}_{top[k].metadata['paragraph_id']}'. "
                try:
                    try:
                        top_content_ += f"'본문': '{top[k].metadata['paragraphs']}'. "            
                    except:
                        top_content_ += f"'본문': '{top[k].page_content}'. "
                except:
                    pass
            
                # 지문 (handout)
                try:
                    top_content_ += f"'지문': '{top[k].metadata['본문']}'. "
                except:
                    top_content_ += f"'지문': '{top[k].page_content}'. "
                
                # 관계 (Relation)
                try:
                    if top[k].metadata['relation'] != 0:
                        top_content_ += f" '관계': '{top[k].metadata['relation']}.'" 
                except:
                    pass
            
            
            elif args.task in [3, 4]:
                try:
                    top_content_ += f"'질문': '{top[k].metadata['질문']}'. "
                except:
                    top_content_ += f"'질문': '{top[k].page_content}'. "
                
                top_content_ += f"'skill': '{top[k].metadata['skill']}'. 'method': '{top[k].metadata['method']}.'"
                
                
            top_content_ += "]"
            
            top_content.append(top_content_)
        top_content = '\n'.join(top_content)
        top_content_list.append(top_content)
        #############################################################################
        #############################################################################
        
        
        #############################################################################    
        ### make every  -------------------------------------------------------------
        #############################################################################
        if args.task == 1:
            # paragraph
            sys_prompt = "'지문'과 '질문'을 보고 아래 '후보' 중 어떠한 '본문'과 가장 높은 관련성을 보이는지 하나 골라 해당 '본문'의 '본문 id'를 답하시오. (이 때, id는 1_1_1_1와 같은 형태이다)"
            user_prompt = f"'지문' : {query_db['본문'][idx]}. '질문' : {query_db['질문'][idx]}.\n"
            user_prompt += f"'후보' : {top_content}.\n"
            user_prompt += "<Paragraph>본문 id<Paragraph> 형태로 답하시오. (예를 들어, <Paragraph>1_1_1_1<Paragraph>)"
        elif args.task == 2:
            # relation
            sys_prompt = "'지문'과 '질문'을 보고 아래 '후보' 중 어떠한 '본문'과 가장 높은 관련성을 보이며 어떠한 관계를 갖는지 '보기' 중에 하나 고르시오."
            sys_prompt += "이 때, 본문의 <underline>이나 <bold> 등이 표시된 부분이 존재한다면 해당 부분을 우선적으로 지문과 비교하시오."
            if args.in_context_sample:
                sys_prompt += "관계를 맞추기 위한 예시는 '예시'에 제공되어 있으며 이를 참고해 답하시오."
            
            user_prompt = f"'지문' : {query_db['본문'][idx]}. '질문' : {query_db['질문'][idx]}.\n"
            user_prompt += f"'후보' : {top_content}.\n"
            if args.in_context_sample:
                # 1.원문 3개
                user_prompt += f"'예시1-1' : 지문 'This is why some scientists believe it is so important to study woodpeckers. They hammer their beaks into trees at speeds of over 20 kilometers per hour. They can peck about 20 times per second. On average, they knock their heads against hard surfaces about 12,000 times every day. Each one of those impacts is about 100 times as powerful as a hit that would cause serious brain injury to a human. Yet somehow, woodpeckers never suffer any physical or mental damage. Why not?'은 본문 'As time passed, numerous innovations were made, making today’s helmets much safer and stronger than Kafka’s original design. They are built to survive massive impacts while remaining light enough for wearers to play sports or do their jobs. Another innovation is that while old-style helmets were heavy and bulky, causing neck pain, today’s helmets are lighter and more comfortable for the wearer. This is important because people are much more likely to wear helmets if they are comfortable.\nDespite all these innovations, helmets are still far from perfect. Sports players as well as workers at construction sites, factories, and other dangerous work environments frequently experience brain injuries due to the force and frequency of blows to the head. Doctors believe that repeated blows to the brain can cause a variety of physical and mental problems later in life.\nThis is why some scientists believe it is so important to study woodpeckers. They hammer their beaks into trees at speeds of over 20 kilometers per hour. They can peck about 20 times per second. On average, they knock their heads against hard surfaces about 12,000 times every day. Each one of those impacts is about 100 times as powerful as a hit that would cause serious brain injury to a human. Yet somehow, woodpeckers never suffer any physical or mental damage. Why not?'와 1. 원문의 관계를 갖는다. 'This is why some scientists believe it is so important to study woodpeckers.' 이후 내용을 본문에서 발췌하여 변형 없이 그대로 사용하였기 때문이다.\n" # handout_id : 877
                user_prompt += f"'예시1-2' : 지문 'Living as an older person was a hard experience for both Regina and Trent, but they consider it an invaluable one. This once-in-a-lifetime opportunity helped them understand not only the physical changes that older people go through but also the way society treats them. By walking in someone else’s shoes, Regina and Trent were able to see that the elderly also enjoy life with passion. Moreover, the experience changed the way they conduct their lives.'은 본문 ' “I realized life was too short to just sit around and wait for things to happen to me,” he said. Now that Trent knows how important it is to plan and save for the future, he has decided to find a more stable job and move out of his parents’ house. Trent has also started to exercise regularly so that he can stay healthy and fully enjoy his golden years in the future.\n Living as an older person was a hard experience for both Regina and Trent, but they consider it an invaluable one. This once-in-a-lifetime opportunity helped them understand not only the physical changes that older people go through but also the way society treats them. By walking in someone else’s shoes, Regina and Trent were able to see that the elderly also enjoy life with passion. Moreover, the experience changed the way they conduct their lives. They hope that this documentary will help raise awareness of the problems the elderly continue to face and help young people have a more positive view of growing older.'와 1.원문의 관계를 갖는다. 본문 중간에서 지문에 해당하는 내용을 일체의 변형 없이 발췌하여 그대로 사용하였기 때문이다.\n" # handout_id : 3410
                user_prompt += f"'예시1-3' : 지문 'First of all, I suggest that you replace sugary drinks such as soft drinks and juice with water. This will reduce your sugar intake and help you to feel full. You can also increase your water intake by eating more fruits and vegetables. Because these foods contain a great deal of water, they can provide up to 20% of the water your body needs each day. In case you get thirsty between meals, you can carry a water bottle with you. You can also flavor your water with fruits or herbs to enjoy it more. Remember, drinking lots of water will help you look and feel better.'은 본문 'Two Drink Well Edward\n Hello, I’m Edward and I’m a nutritionist. Let me ask you a question. This special drink will help you reduce stress, increase energy, and maintain a healthy body weight. What drink am I talking about? In fact, this magical drink is something that you all know. It’s water! Do you also want to have nice skin? Drink water. Water is nature’s own beauty cream. Drinking water hydrates skin cells, giving your skin a healthy glow. Likewise, water is very important for basic body functions because about 70% of our body is water, and we need about 2 liters of water a day. However, many of us don’t get enough water and eventually experience dehydration. For this reason we have to drink plenty of water.\n So how can we increase our water intake? First of all, I suggest that you replace sugary drinks such as soft drinks and juice with water. This will reduce your sugar intake and help you to feel full. You can also increase your water intake by eating more fruits and vegetables. Because these foods contain a great deal of water, they can provide up to 20% of the water your body needs each day. In case you get thirsty between meals, you can carry a water bottle with you. You can also flavor your water with fruits or herbs to enjoy it more. Remember, drinking lots of water will help you look and feel better.'와 1. 원문의 관계를 갖는다. 'First of all, I suggest that you replace sugary drinks such as soft drinks and juice with water.' 이후 내용을 본문에서 변형 없이 발췌하여 그대로 사용하였기 때문이다.\n"  # handout_id : 177
                # 2. 삭제 3개
                user_prompt += f"'예시2-1' : 지문 'Until 1966, no one knew that the Mugujeonggwang Daedaranigyeong, the world's oldest printed document, lay inside a container at Bulguksa Temple in Gyeongju, Korea. Experts around the world were shocked that a document printed more than 1,200 years ago could still be around. They were even surprised when the paper was removed from the container. Although the document was printed before 751 CE, it was still in perfect condition. This discovery proved that the paper-making technology of the Unified Silla Kingdom era. (676-935) was more advanced than that of either Japan or China, both of which also had highly developed paper-making technology.'은 본문 'Until 1966, no one knew that the Mugujeonggwang Daedaranigyeong, the world’s oldest printed document, lay inside a container at Bulguksa Temple in Gyeongju, Korea. Experts around the world were shocked that a document printed more than 1,200 years ago could still be around. They were even more surprised when the paper was removed from the container. Although the document was printed before 751 CE, it was still in perfect condition. \nThis discovery proved that the paper-making technology of the Unified Silla Kingdom era (676–935) was more advanced than that of either Japan or China, both of which also had highly developed paper-making technology. How could this paper last for more than 1,000 years without breaking down or becoming damaged? The secret lies in hanji’s amazing physical properties.'와 2.삭제의 관계가 있다. 본문 중간의 'They were even more surprised when the paper was removed from the container.' 문장에서 'more' 단어를 삭제하고 지문으로 사용하였기 때문이다.\n" #handout_id: 1674
                user_prompt += f"'예시2-2' : 지문 'One of hanji's newest uses is a <underline>(A) t___</underline> for the ears. Customers can now buy speakers that use vibration plates and outside panels made of hanji. Compared to regular speakers, the sound that comes from hanji speakers is stronger and sharper. The paper's thickness and ability to absorb sound help the speakers pick up the smallest vibrations. In addition, the fact that the sound will not change over time because of the strength of hanji makes these speakers a great <underline>(B) p___</underline>. Serious music lovers will really be able to appreciate the great sound quality of these speakers.'은 본문 'Lately, designers have been using hanji to make clothes, socks, and ties. The fabric these designers are using is a blend of hanji yarn with cotton or silk. This blend is almost weightless and keeps its shape better than other materials. It is also washable and eco-friendly. Not only is hanji clothing practical, but it’s also making waves at domestic and international fashion shows. It seems that hanji clothing is here to stay. \nOne of hanji’s newest uses is a treat for the ears. Customers can now buy speakers that use vibration plates and outside panels made of hanji. Compared to regular speakers, the sound that comes from hanji speakers is stronger and sharper. The paper’s thickness and ability to absorb sound help the speakers pick up the smallest vibrations. In addition, the fact that the sound will not change over time because of the strength of hanji makes these speakers a great purchase. Serious music lovers will really be able to appreciate the great sound quality of these speakers.'와 2.삭제의 관계가 있다. 본문의 'One of hanji’s newest uses is a treat for the ears.' 문장에서 'treat' 단어의 일부가 삭제되었으며, 본문의 'In addition, the fact that the sound will not change over time because of the strength of hanji makes these speakers a great purchase.' 문장에서 'purchase' 단어의 일부가 삭제되어 지문으로 사용되었기 때문이다.\n"  # handout_db : 1674
                user_prompt += f"'예시2-3' : 지문 'Making the decision to be green is not really a big one. <bold>(①)</bold> It is not difficult.\xa0<bold>(②)</bold> Some people think having a green wardrobe is going to cost them more money or be too much trouble. <bold>(③)</bold> You may already have shared clothes with your friends or given your old clothes to charity. <bold>(④)</bold> Or possibly you have reused clothes instead of throwing them out. <bold>(⑤)</bold> Just add 'Reduce' to your going green list, and you will <underline>___</underline>.'은 본문 '6. Making the decision to be green is not really a big one. It is not difficult. Some people think having a green wardrobe is going to cost them more money or be too much trouble. However, chances are that you are already greener than you think. You may already have shared clothes with your friends or given your old clothes to charity. Or possibly you have reused clothes instead of throwing them out. Just add ‘Reduce’ to your going green list, and you will make a real difference to the environment.\n7. Once you start to go green, you will find lots of ways in which you can get into the eco-fashion scene. You will also discover how easy and rewarding being green is. Just knowing that you are doing your part to preserve the planet for the future is one of the best feelings ever.\nFamous sayings about the three R’s\n1. Recycle\nOne person’s trash is another’s treasure.\n2. Reuse\nThere is no zero waste without reuse.\n3. Reduce\nThere’s always a way to do without something.'와 2.삭제의 관계가 있다. 본문 중간의 'However, chances are that you are already greener than you think.' 문장이 삭제되었으며, 'Just add ‘Reduce’ to your going green list, and you will make a real difference to the environment.' 문장에서 'environment' 단어가 삭제되어 __으로 나타났기 때문이다.\n" # handout_id : 745
                # 3. 교체 및 삽입 3개
                user_prompt += f"'예시3-1' : 지문 'Without Don’s permit, I would have had to pay a $100,000 fine or ended up in the police station. Soon after dusk, the camp site became very dark. After a barbecue dinner, Maddie and I gazed at the clear sky. The absence of all artificial city lights made the stars in the sky more brilliant and easy to locate. The dense, quiet forest, and rotting logs made it feel as if time had stood still here for centuries. Breaking the silence, Don reminded us not to leave leftover food in the tent because it could attract bears. Imagining a bear lick my face, I got frightened and my romantic night was over. I promptly fled to my tent.'은 본문 ' Upon arriving at the camp site, next to a small creek Maddie’s parents started setting up the tent. So, Maddie and I went to the beach and collected shellfish. I could’ve filled a basket with the shellfish but Don, Maddie’s dad, advised us not to. Don explained that people need a legitimate National Park fishing permit in order to fish or collect shellfish in all Canadian National Parks.\n However, those who are under 16 do not need to obtain the permit as long as they are accompanied by an adult who has one. He also told us that people caught taking undersized seafood have their seafood seized and are fined $100,000 by the judicial system. I was so surprised by the size of the fine that I tipped out my basket in the weeds right away. Without Don’s permit, I would have had to pay a $100,000 fine or ended up in the police station.\n Soon after dusk, the camp site became very dark. After a barbecue dinner, Maddie and I gazed at the clear sky. The absence of all artificial city lights made the stars in the sky more brilliant and easy to locate. The dense, quiet forest, and rotting logs made it feel as if time had stood still here for centuries. Breaking the silence, Don reminded us not to leave leftover food in the tent because it could attract bears. Imagining a bear licking my face, I got frightened and my romantic night was over. I promptly fled to my tent.'와 3.교체 및 삽입의 관계가 있다. 'Imagining a bear licking my face, I got frightened and my romantic night was over. I promptly fled to my tent.'의 'licking' 단어가 'lick'으로 교체되었기 때문이다.\n" # handout_id:20125
                user_prompt += f"'예시3-2' : 지문 '1. You probably know of great souls who sacrificed <bold>29) [ them / themselves ]</bold> to help others and <bold>30) [ make / making ]</bold> the world a better place to <bold>31) [ live / live in ]</bold>. It may seem <bold>32) [ difficult / difficultly ]</bold> or practically impossible for <bold>33) [ ordinary / ordinarily ]</bold> people to live up to what Dr. Schweitzer did. But small actions <bold>34) [ that / with which ]</bold> we take for our family and friends in our everyday lives can make a difference toward <bold>35) [ create / creating ]</bold> a better world. Today we are going to listen to the stories of two teenagers who have <bold>36) [ taken / been taken ]</bold> such actions.'은 본문 'You probably know of great souls who sacrificed themselves to help others and make the world a better place to live in. It may seem difficult or practically impossible for ordinary people to live up to what Dr. Schweitzer did. But small actions that we take for our family and friends in our everyday lives can make a difference toward creating a better world. Today we are going to listen to the stories of two teenagers who have taken such actions.\nSpreading Kindness with Positive Messages Annie from Ottawa\nHi, everyone. Nice to meet you all here today. I’m Annie from Ottawa. You know what these yellow sticky notes are for and probably use them for many purposes. I am here to tell you how I use them. It’s to encourage people, give them strength, and help them feel happy. When I was in middle school, someone broke into my locker and used my smartphone to post hateful things on my SNS page. It was so hurtful and difficult to overcome. But after a lot of thinking and talking with my parents and closest friends, I concluded that although bullies use words to hurt people, I should use them to encourage others.'와 3.교체 및 삽입 관계가 있다. 'You probably know of great souls who sacrificed themselves to help others and make the world a better place to live in.'의 문장에 'them', 'making', 'live'가 선택지로 삽입되었고, 'It may seem difficult or practically impossible for ordinary people to live up to what Dr. Schweitzer did.'의 문장에 'difficultly', 'ordinarily'가 선택지로 삽입되었고, 'But small actions that we take for our family and friends in our everyday lives can make a difference toward creating a better world.' 문장에 'with which', 'create'가 선택지로 삽입되었으며, 'Today we are going to listen to the stories of two teenagers who have taken such actions.' 문장에 'been taken'이 선택지로 삽입되었기 때문이다.\n" # handout_id:3127
                user_prompt += f"'예시3-3' : 지문 '1. You probably know of great souls who <underline>123) were sacrificed</underline> themselves to help others and make the world a better place to <underline>124) live</underline>. It may seem <underline>125) difficultly</underline> or <underline>126) practical</underline> impossible for ordinary people to live up to what Dr. Schweitzer did. But small actions <underline>127) where</underline> we take for our family and friends in our everyday lives can make a <underline>128) different</underline> toward creating a better world. Today we are going to <underline>129) listening</underline> to the stories of two teenagers <underline>130) have</underline> taken such actions.'은 본문 'You probably know of great souls who sacrificed themselves to help others and make the world a better place to live in. It may seem difficult or practically impossible for ordinary people to live up to what Dr. Schweitzer did. But small actions that we take for our family and friends in our everyday lives can make a difference toward creating a better world. Today we are going to listen to the stories of two teenagers who have taken such actions.\nSpreading Kindness with Positive Messages Annie from Ottawa\nHi, everyone. Nice to meet you all here today. I’m Annie from Ottawa. You know what these yellow sticky notes are for and probably use them for many purposes. I am here to tell you how I use them. It’s to encourage people, give them strength, and help them feel happy. When I was in middle school, someone broke into my locker and used my smartphone to post hateful things on my SNS page. It was so hurtful and difficult to overcome. But after a lot of thinking and talking with my parents and closest friends, I concluded that although bullies use words to hurt people, I should use them to encourage others.'와 3.교체 및 삽입의 관계를 갖는다. 'You probably know of great souls who sacrificed themselves to help others and make the world a better place to live in.' 문장에서 'sacrificed'를 'were scarificed', 'live in'을 'live'로 교체하였으며, 'It may seem difficult or practically impossible for ordinary people to live up to what Dr. Schweitzer did.' 문장에서 'difficult'를 'difficulty'로, 'practically'를 'practical'로 교체하였고, 'But small actions that we take for our family and friends in our everyday lives can make a difference toward creating a better world.' 문장에서 'that'을 'where'로, 'difference'를 'different'로 교체하였으며, 'Today we are going to listen to the stories of two teenagers who have taken such actions.' 문장에서 'listen'을 'listening'으로 교체하였기 때문이다.\n" # handout_id:3226
                # 4. 복합 2개
                user_prompt += f"'예시4-1' : 지문 'As time passed, numerous innovations <underline>① were made</underline>, <underline>🅐 making</underline> today’s helmets <underline>⒜___</underline> safer and stronger than Kafka’s original design. They <underline>② built</underline> to survive massive impacts while remaining light enough for wearers to play sports or <underline>③ do</underline> their jobs. Another innovation is that <bold>⑴ [ while / as ]</bold> old-style helmets were heavy and bulky, <underline>🅑 causing</underline> neck pain, today’s helmets are lighter and more comfortable for the wearer.'은 본문 'As time passed, numerous innovations were made, making today’s helmets much safer and stronger than Kafka’s original design. They are built to survive massive impacts while remaining light enough for wearers to play sports or do their jobs. Another innovation is that while old-style helmets were heavy and bulky, causing neck pain, today’s helmets are lighter and more comfortable for the wearer.'과 4.복합 관계를 갖는다. 'As time passed, numerous innovations were made, making today’s helmets much safer and stronger than Kafka’s original design.' 문장에서 'much'가 삭제되었으며,  'They are built to survive massive impacts while remaining light enough for wearers to play sports or do their jobs.' 문장에서 'are built'가 'built'로 교체되었으며, 'Another innovation is that while old-style helmets were heavy and bulky, causing neck pain, today’s helmets are lighter and more comfortable for the wearer.' 문장에서 'as'가 선택지로 추가되었기 때문에, 2.삭제와 3.교체 및 삽입 관계를 동시에 갖는다. 따라서 주어진 본문과 지문은 4.복합 관계를 갖는다.\n" # handout_id : 853
                user_prompt += f"'예시4-2' : 지문 'Drinking water hydrates skin cells, giving your skin a healthy glow. Likewise, water is very important for basic body functions because about 70% of our body is water, and we need about 2 liters of water a day. However, many of us don’t get enough water and eventually suffer <underline>___</underline>. For this reason we have to drink plenty of water. So how can we increase our water intake? First of all, I suggest that you replace sugary drinks such as soft drinks and juice with water.'은 본문 'Drinking water hydrates skin cells, giving your skin a healthy glow. Likewise, water is very important for basic body functions because about 70% of our body is water, and we need about 2 liters of water a day. However, many of us don’t get enough water and eventually experience dehydration. For this reason we have to drink plenty of water.\n So how can we increase our water intake? First of all, I suggest that you replace sugary drinks such as soft drinks and juice with water.'와 4.복합 관계를 갖는다. 'However, many of us don’t get enough water and eventually experience dehydration.' 문장에서 'experience'는 'suffer'로 교체되었고, 'dehydration'은 삭제되었다. 즉, 2.삭제와 3.교체 및 삽입 관계를 동시에 갖기 때문에 해당 본문과 지문은 4.복합의 관계를 갖는다.\n" # handout_id:22
        
            user_prompt += "'보기' : [1: 원문 (본문의 일부를 변형없이 발췌 혹은 본문 전체를 그대로 지문으로 사용), 2: 삭제 (본문에서 특정 단어/문장을 삭제하여 지문으로 사용), 3: 교체 및 삽입 (본문에 없던 단어/문장을 추가하여 지문으로 사용 ), 4. 복합 (원문, 삭제, 삽입 관계가 복합적으로 적용)].\n"
            user_prompt += "<Relation>관계(int)<Relation> 형태로 답하고 (예를 들어, <Relation>1<Relation>), 해당 관계를 고른 이유를 <Description>관계정보를 고른 이유<Description> 형태로 자세히 서술하시오."
        elif args.task == 3:
            # skill : 문제를 풀기 위해 필요한 능력
            sys_prompt = "'참고'를 참고하여 '질문'을 보고 이 문제를 풀기 위한 능력을 '보기' 중에 하나 고르시오."
            user_prompt = f"'질문' : {query_db['질문'][idx]}.\n"
            user_prompt += f"'참고' : {top_content}.\n"
            user_prompt += f"'보기' : [101: 어휘 뜻 이해 (어휘의 뜻을 이해한다.), 102: 영영 풀이 (어휘의 뜻을 영어로 이해한다.), 103: 어휘 혼합, 201: 용법 이해 (용법을 이해한다.), 202: 용법일치불일치 판단 (용법이 서로 같은지 다른지 판단한다.), 203: 문법 혼합, 301: 목적 이해 (글의 목적을 이해한다.), 302: 주제 이해 (글의 주제를 이해한다.), 303: 제목 이해 (글의 제목을 이해한다.), 304: 주장 이해 (글의 주장을 이해한다.), 305: 요지 이해 (글의 요지를 이해한다.), 306: 의미 이해 (글의 의미를 이해한다.), 307: 분위기 이해 (글의 분위기를 이해한다.), 308: 심경 이해 (글의 화자의 심경을 이해한다.), 309: 심경 변화 이해 (글의 화자의 심경 변화를 이해한다.), 310: 어조 이해 (글의 어조를 이해한다.), 311: 순서 이해 (글의 내용을 이해한다.), 312: 대상 이해 (지칭하는 대상을 이해한다), 313: 내용이해 혼합, 401: 내용유추 (글의 내용을 유추한다.), 402: 순서유추 (글의 순서를 유추한다.), 403: 어휘유추 (특정 위치의 어휘를 유추한다.), 404: 연결어유추 (특정 위치의 연결어를 유추한다.), 405: 지칭유추 (지칭하는 대상을 유추한다.), 406: 어휘유추 전반 (유추 내용을 복합적으로 묻는 경우), 407: 내용일치불일치 판단 (내용이 서로 같은지 다른지 판단한다.), 408: 요약 (글을 요약한다.), 409: 번역 (글을 한글로 변역한다.), 410: 영작 (글을 영어로 작문한다.), 411: 내용응용 혼합, 501: 영역통합, 601: 기타].\n"
            user_prompt += f"<Skill>정답<Skill> 형태로 답하라. (예시, <Skill>405<Skill>)"
        elif args.task == 4:
            # method : 해당 문제의 '질문'이 학습자의 역량을 검증하기 위해 어떤 방식으로 질문하는지를 의미
            sys_prompt = "'참고'를 참고하여 '질문'을 보고 해당 문제가 학습자의 역량을 검증하기 위해 어떠한 방식으로 질문하는지 '보기' 중에 하나 고르시오."
            user_prompt = f"'질문' : {query_db['질문'][idx]}.\n"
            user_prompt += f"'참고' : {top_content}.\n"
            user_prompt += f"'보기' : [1: 맞는 것 찾기(단수) (맞는 것을 찾는다.), 2: 맞는 것 찾기(복수) (맞는 것을 모두 찾는다.), 3: 맞는 것 세기(개수) (맞는 것을 찾아서 개수를 센다.), 4: 틀린 것 찾기(단수) (틀린 것을 찾는다.), 5: 틀린 것 찾기(복수) (틀린 것을 모두 찾는다.), 6: 틀린 것 세기(개수) (틀린 것을 찾아서 개수를 센다.), 7: 다른 것 찾기 (다른 것을 찾는다.), 8: 맞는 위치 찾기 (맞는 위치를 찾는다.), 9: 바른 배열 찾기 (맞는 배열을 찾는다.), 10: 바른 조합 찾기 (맞는 조합을 찾는다.), 11: 어휘 쓰기(보기에서 골라) (맞는 어휘를 보기에서 찾아 쓴다.), 12: 어휘 쓰기(본문에서 찾아) (맞는 어휘를 본문에서 찾아 쓴다.), 13: 어휘 쓰기(고쳐/직접) (맞는 어휘로 고쳐쓰거나 직접쓴다.), 14: 문장 쓰기 (문장을 쓴다.), 15: 바른 배열 쓰기 (맞는 배열하여 쓴다.), 16: 혼합, 17: 기타].\n"
            user_prompt += f"<Method>정답<Method> 형태로 답하라. (예시, <Method>5<Method>)"
        #############################################################################
        #############################################################################
        
                                 
        task = {
                "custom_id": f"task-{idx}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": args.llm_model,
                    "temperature": args.temperature,
                    ## incompatible with "gpt-4o"
                    # "response_format": { 
                    #     "type": "json_object"
                    # },
                    "messages": [
                        {
                            "role": "system",
                            "content": sys_prompt
                        },
                        {
                            "role": "user",
                            "content": user_prompt
                        }
                    ],
                }
            }
        tasks.append(task)
    
    ## create batch file (json type)
    file_name = osp.join(args.result_path, f"batch_tasks.jsonl")
    with open(file_name, 'w') as file:
        for obj in tasks:
            file.write(json.dumps(obj) + '\n')

    ## uploading batch file
    batch_file = client.files.create(
                file=open(file_name, "rb"),
                purpose="batch"
                )

    ## creating the batch job
    batch_job = client.batches.create(
                input_file_id=batch_file.id,
                endpoint="/v1/chat/completions",
                completion_window="24h"
                )
    print(f"Batch API job ID : {batch_job.id}")
    if not args.ignore_wandb:
        wandb.config.update({f"batch_job_id": batch_job.id})
        
    ## checking batch status
    while True:
        batch_job = client.batches.retrieve(batch_job.id)
        print(f"Batch API status : {batch_job.status}")
        if batch_job.status in ['failed', "cancelling", "cancelled", "expired"]:
            raise ValueError("Fail to send batch api:(")
        elif batch_job.status in ["finalizing"]:
            print("Waiting for result being prepared")
        elif batch_job.status in ["completed"]:
            print(f"Completed batch job!")
            break
        time.sleep(30)
        
    ## retrieving results and save as jsonl file
    result_file_id = batch_job.output_file_id
    result = client.files.content(result_file_id).content
    result_file_name = osp.join(args.result_path, f"batch_job_results.jsonl")
    with open(result_file_name, 'wb') as file:
        file.write(result)
    
    with open(result_file_name, 'wb') as file:
        file.write(result)
        ## Loading data from saved file (json)
        results = []
        with open(result_file_name, 'r') as file:
            for line in file:
                # Parsing the JSON string into a dict and appending to the list of results
                json_object = json.loads(line.strip())
                results.append(json_object)
                    
        for i, res in enumerate(results):
            task_id = res['custom_id']
            # Getting index from task id
            index = task_id.split('-')[-1]
            
            # get index in query_db
            query_idx = query_db.iloc[int(index)]['handout_id']
            idx_list.append(query_idx)
            
            answer = res['response']['body']['choices'][0]['message']['content']
            # print(f"RESULT : batch:{batch}/{num_batches}. idx:{start_idx+i}. \n {answer}")
            print(f"RESULT {i} : {answer}")
            answer_list.append(answer)
            
            tmp_dict = extract_label(answer)
            id_list.append(tmp_dict['paragraph'])
            try:
                textbook_list.append(float(tmp_dict['paragraph'].split('_')[0]))
                unit_list.append(float(tmp_dict['paragraph'].split('_')[1]))
                story_list.append(float(tmp_dict['paragraph'].split('_')[2]))
                paragraph_list.append(float(tmp_dict['paragraph'].split('_')[3]))
            except:
                textbook_list.append(-9999)
                unit_list.append(-9999)
                story_list.append(-9999)
                paragraph_list.append(-9999)
                
            skill_list.append(tmp_dict['skill'])
            method_list.append(tmp_dict['method'])
            relation_list.append(tmp_dict['relation'])
            try:
                description_list.append(tmp_dict['description'])
            except:
                description_list.append(-9999)
            print("\n\n----------------------------\n\n")
        
        
    ## save_result
    try:
        label_df = pd.DataFrame({"query_idx" : idx_list,
                                "id" : id_list,
                                "textbook_id" : textbook_list,
                                "unit_id" : unit_list,
                                "story_id" : story_list,
                                "paragraph_id" : paragraph_list,
                                "skill" : skill_list,
                                "method" : method_list,
                                "relation" : relation_list,
                                "description" : description_list,
                                },)
        label_df.to_csv(osp.join(args.result_path, 'answer_df.csv'),
                                encoding="utf-8-sig", index=False)
    except:
        print("Problem with saving label df")

    with open(osp.join(args.result_path, 'answer_list.json'), 'w') as json_file:
        json.dump(answer_list, json_file, ensure_ascii=False)
        
    with open(osp.join(args.result_path, 'top_k_list.json'), 'w') as json_file:
        json.dump(top_content_list, json_file, ensure_ascii=False)
    
    print(f"Save result(answer and top_k) in {args.result_path}")




if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--openai_api_key',type=str, default=None, required=True)
    
    parser.add_argument('--query_path', type=str, default="./data/solvook_handout_te.csv")
    parser.add_argument('--vector_db_path', type=str, default="./data/vector_db.json")
    
    parser.add_argument('--llm_model', type=str, default='gpt-4o')
    parser.add_argument('--temperature', type=float, default=0.0)
    
    parser.add_argument('--task', type=int, choices=[1, 2, 3, 4], default=1,
                        help="1: paragraph, 2: relation, 3: skill, 4: method")
    parser.add_argument('--in_context_sample', action='store_true', default=False,
                        help='in case of task no.2, adopt in-context sample, not retrieved sample')
        
    parser.add_argument('--result_path', type=str, default='./exp_result')
    
    parser.add_argument('--only_train', action='store_true', default=False)
    
    ## wandb
    parser.add_argument("--ignore_wandb", action='store_true', default=False)
    parser.add_argument("--wandb_project", type=str, default="tips_2024")
    parser.add_argument("--wandb_entity", type=str, default="sungjun98")

    
    args = parser.parse_args()
    
    ## set save_path
    save_name = f"seed_{args.seed}/task_{args.task}/{args.llm_model}"
    args.result_path = osp.join(args.result_path, save_name)
    print(f"Set save path on {args.result_path}")
    os.makedirs(args.result_path, exist_ok=True)
    
    
    ## set wandb
    if not args.ignore_wandb:
        wandb.init(project=args.wandb_project, entity=args.wandb_entity)
        wandb.config.update(args)
        wandb.run.name = save_name
    
    ### [Step 1] Load Data and Make Loader ##embeddings as unique 본문, 질문, paragraphs
    print("[Step 1] Load Data!!")
    vector_db, query_db = load_solvook_data(args)
    
    args_dict=vars(args)
    with open(osp.join(args.result_path, 'args.json'), 'w') as json_file:
        json.dump(args_dict, json_file, ensure_ascii=False)
    
    print("[Step 2] Start generation...")
    generation(args, vector_db, query_db)
    print("Finally End generation!!")
    
    
    if not args.only_train:
        print("[Step 3] Evaluate")
        from eval import calculate_paragraph_acc, calculate_method_acc, calculate_skill_acc, calculate_relation_acc
        answer_db = pd.read_csv(osp.join(args.result_path, "answer_df.csv"))
        
        textbook_cor, story_cor, unit_cor, parap_cor,total_parap_cor, total_parap_acc = calculate_paragraph_acc(answer_db, query_db)
        skill_cor, skill_acc = calculate_skill_acc(answer_db, query_db)
        method_cor, method_acc = calculate_method_acc(answer_db, query_db)
        relation_cor, relation_acc = calculate_relation_acc(answer_db, query_db)
        
        if not args.ignore_wandb:
            wandb.run.summary['textbook_cor'] = sum(textbook_cor)
            wandb.run.summary['story_cor'] = sum(story_cor)
            wandb.run.summary['unit_cor'] = sum(unit_cor)
            wandb.run.summary['parap_cor'] = sum(parap_cor)
            wandb.run.summary['total_parap_cor'] = total_parap_cor
            wandb.run.summary['parap acc.'] = total_parap_acc
            
            wandb.run.summary['skill_cor'] = skill_cor
            wandb.run.summary['skill acc.'] = skill_acc
            
            wandb.run.summary['method_cor'] = method_cor
            wandb.run.summary['method acc.'] = method_acc
            
            wandb.run.summary['relation_cor'] = relation_cor
            wandb.run.summary['relation acc.'] = relation_acc