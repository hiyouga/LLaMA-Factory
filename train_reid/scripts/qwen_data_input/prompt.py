prompt = """
###Task###
You are an expert in the field of Re-Identification (ReID). Please refer to the following requirements to understand the general and specific features of the image, and then combine all the features to return a descriptive language.
###Requirement###
Choose one color from black, white, red, purple, yellow, blue, green, pink, gray, and brown.
1、Gender: male、female
2、Age: teenager、young、adult、old
3、Body Build: fat、slightly fat、thin
4、Length Hair: long hair、medium-length hair、short hair、bald
5、Wearing hat: yes、no; if yes, the color is: XXX
6、Carrying backpack: yes、no; if yes, the color is:  XXX
7、Carrying handbag or bag: yes、no; if yes, the color is:  XXX
8、Upper Body
8.1、Sleeve Length: long sleeve、short sleeve
8.2、Inner Lining: yes、no; if yes, the color is:
8.3、Color of upper-body: XXX
9、Lower Body
9.1、Length of lower-body: long lower-body clothing、short
9.2、Type of lower-body: dress、pants
9.3、Color of lower-body: XXX
10、Shoe Color: XXX
11、Emotion: Happy、Surprised、Sad、Angry、Disgusted、Fearful、Neutral、Other
12、Gait and Posture: XXX
###Output###
Combine all the attributes above into a natural language as the final output.
"""