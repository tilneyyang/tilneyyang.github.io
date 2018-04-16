---
layout: default
---
# Conversational Agents Basics

一个典型的对话系统架构如下图所示

![conversational agent architecture](/assets/img/conversational-agents.png)

系统主要由6个组件构成. 其中Speech Recognition(ASR)和Natural Language Understanding(NLU)主要是从用户的输入中抽取语义信息, 而Natural Language Generation(NLG)和Text-to-Speech Synthesis(TTS)模块则是将语义信息转换成回答. Dialogue Manager控制整个对话过程, Task Manager则维护具体领域的相关知识(如在一个天气机器人中维护所有地点的天气信息). 接下来, 我们会大体过一遍这些模块以及其负责的事情, 考虑到之后的场景处理的更多是文本输入和文本输出, 将会跳过ASR和TTS模块.

## NLU
The NLU (natural language understanding) component of dialogue systems must produce a semantic representation which is appropriate for the dialogue task. The main purpose of task-oriented dialogue systems is to get the information (parameters) needed to perform this task. So the most common used representation method for them is frame-and-slot semantics. The frame with slots of a travel system is shown as follow:

```python
ticket:
  flight:
    origin:
        city: Boston
        date:            day-of-week:  Tuesday => normalize to 2016/11/07         TIME:            part-of-day:  morning => normalize to [8:00 to 12:59]
    dest:
        city: San Francisco
```

生成Frame-and-slot semantics的方法有很多, 如基于semantic grammar, semantic HMM等.

## Response Generation
Response Generation模块选择要表达给用户的概念(concepts), 并且计划用文本表达这些概念的表达方式. Generation task主要可以分为两个任务:*what to say*, 和*how to say it*. 

### Content Planer
主要针对地一个问题, 决定最终展示给用户的内容, 如是问一个问题,还是直接给出一个答案. Content planning模块一般都被合并到Dialogue manager中. For a system with Frame-and-slot semantics, the content planer will continously return question for the unfilled slots. A response is generated only when all the information needed for the task to perform is gathered

### Language Generation
Language generation模块则是针对第二个任务,选择合适的句法结构和词语来表达所需要表达的意思. 
The most simple way for language generation is template-based generation. Dialogue designer prespecified all or most of the words in the sentence to be uttered to the user. Sentences created from these templates are often called **prompts**

```
What time do you want to leave CITY-ORIG??
Will you return to CITY-ORIG from CITY-DEST?
```

A second method for language generation relies on techniques from the field nat- ural language generation. Here the dialogue manager builds a representation of the meaning of the utterance to be expressed, and passes this meaning representation to a full generator.

## Dialogue Manager
The final component of a dialogue system is the dialogue manager, which controls the architecture and structure of the dialogue. The dialogue manager takes input from the ASR/NLU components, maintains some sort of state, interfaces with the task manager, and passes output to the NLG/TTS modules.

### Initiative
We say that the speaker that is in control of the conversation has the initiative; in normal human-human dialogue, initiative shifts back and forth between the participants.

  * **System Initiative**: system completely controls the conversation with the user. It asks the user a series of questions, ignoring (or misinterpreting) anything the user says that is not a direct answer to the system’s question, and then going on to the next question. Dialogue sytems with finite-state manager are pure system initiative. Most of the finite-state based systems allow **universal** commands. **Universals** are commands that can be said anywhere in the dialogue; every dialogue state recognizes the universal commands in addition to the answer to the question that the system just asked.
  * **User Initiative**: user has all the controll of the conversation. Pure user initiative systems are generally used for stateless database querying systems, where the user asks single questions of the system, which the system converts into SQL database queries, and returns the results from some database.
  * **Mixed Initiative**: conversational initiative can shift between the system and user at various points in the dialogue.  **Frame-based** or **form-based** dialogue managers(manager with Frame-and-slot semantics) asks the user questions to fill slots in the frame, but allow the user to guide the dialogue by giving information that fills other slots in the frame. Since users may switch from frame to frame, the system must be able to disambiguate which slot of which frame a given input is supposed to fill, and then switch dialogue control to that frame.

### Prompt Type
  * An **open prompt** is one in which the system gives the user very few constraints, allowing the user to respond however they please.
  * A **directive prompt** is one which explicitly instructs the user how to respond: *Say yes if you accept the call; otherwise, say no.*
  
  

 |  | |Prompt Type|
 | ------ | ------ | ------ |
 | Grammar| Open| Direct |
 |Restrictive| *don't make scense*|System Initiative|
 |Non-restrictive|User Initiative| Mixed Initiative| 
                                                                                                                                                                                                      

Table 1: Operational definition of initiative


### Error Handling: Confirmation/RejectionIn a dialogue system, mishearings are a particularly important class of problems, because speech recognition has such a high error rate. It is therefore important for dialogue systems to make sure that they have achieved the correct interpretation of the user’s input. This is generally done by two methods: **confirming** understandings with the user, and **rejecting** utterances that the system is likely to have misunderstood.


## Dialogue System Evaluation
user testing and evaluation is crucial in dialogue system design. Computing a user satisfaction rating can be done by having users interact with a dialogue system to perform a task, and then having them complete a questionnaire. For example Fig. 24.14 shows multiple-choice questions adapted from Walker et al. (2001); responses are mapped into the range of 1 to 5, and then averaged over all questions to get a total user satisfaction rating.
However it is often economically infeasible to run complete user satisfaction studies after every change in a system. For this reason it is often useful to have performance evalua- tion heuristics which correlate well with human satisfaction

  * **Task Completion Success**: Task success can be measured by evaluating the correctness of the total solution. For a frame-based architecture, this might be the percentage of slots that were filled with the correct values, or the percentage of subtasks that were completed.
  * **Efficiency Cost**: Efficiency costs are measures of the system’s efficiency at helping users. This can be measured via the total elapsed time for the dialogue in seconds, the number of total turns or of system turns, or the total number of queries (Polifroni et al., 1992). Other metrics include the number of system non-responses, and the “turn correction ratio”: the number of system or user turns that were used solely to correct errors, divided by the total number of turns (Danieli and Gerbino, 1995; Hirschman and Pao, 1993).
  * **Quality Cost**: Quality cost measures other aspects of the interaction that affect users’ perception of the system. One such measure is the number of times the ASR system failed to return any sentence, or the number of ASR rejection prompts. Similar metrics include the number of times the user had to **barge-in** (interrupt the system), or the number of time-out prompts played when the user didn’t respond quickly enough. Other quality metrics focus on how well the system understood and responded to the user. This can include the inappropriateness (verbose or ambiguous) of the system’s questions, answers, and error messages (Zue et al., 1989), or the correctness of each question, answer, or error message (Zue et al., 1989; Polifroni et al., 1992).  A very important quality cost is **concept accuracy** or **concept error rate**, which measures the percentage of semantic concepts that the NLU component returns correctly. For frame-based architectures this can be measured by counting the percentage of slots that are filled with the correct meaning. For example if the sentence ‘I want to arrive in Austin at 5:00’ is misrecognized to have the semantics ”DEST-CITY: Boston, Time: 5:00” the concept accuracy would be 50% (one of two slots are wrong).

 We can use additional models to compute weight of these metrics to combine them in evaluation process.