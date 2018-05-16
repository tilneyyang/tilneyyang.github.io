---
layout: default
---
# Dialog Management: A Review
Modern task-oriented dialog systems usually try to help users execute some database action using natural language. This post mainly focuses on ** dialog management** of a task-oriented dialog system. Details of complete task oriented dialog system could be found [here]({{ site.baseurl }}{% post_url 2016-11-02-conversational-agents-basics %})

## Definition
dialog Manager(DM) controls the architecture and structure of the dialog, and is responsible for the state and flow of the conversation. It takes input from the ASR/NLU components, maintains some sort of state(dialog state), interfaces with the task manager and then passes output to the NLG/TTS modules.  
This definition could be a little bit abstract, too many questions like what does the input from NLU look like? what does the maintained state look like? etc. are still here, but we will get to the answers soon.

### dialog State
A dialog state $$s_t$$ is a data structure drawn from a set $$S$$ that summarizes the dialog history up to time $$t$$ to a level of detail that provides sufficient information for choosing the next system action. In practice, the dialog state typically encodes the user’s goal
in the conversation along with relevant history.  
For a frame-based system, A dialog state usually consists of following three components: goals, method. requested slots.

#### Goals
The goal at a given turn is the user’s true required value for each slot in the ontology as has been revealed in the dialog up until that turn. If a slot has not yet been informed by the user, then the goal for that slot is ‘None’. 

#### Method
The ‘method’ of a user’s turn describes the way the user is trying to interact
with the system,  and is one of the following values: by name, by constraints, by alternatives or finished. The method may be inferred if we know the true user’s action using the following rules:

1. The method becomes ‘by constraints’ if the user gives a constraint by
specifying a goal for a particular slot. E.g. inform(food=chinese)
2. The method becomes ‘by alternatives’ if the user issues a ‘reqalts’(request alternatives) act. E.g. 'Do you have any other recommendation other than this?'
3. The method becomes ‘by name’ if the user either informs a value for the
name slot, i.e. the user requests information about a specific named venue. The method becomes 'by name' when system helps helps user find some good candidate, and then user ask question about this specific candidate. 'By constraints' usually returns multiple candidates info, in contrast 'by name' only return detailed information of some exact candidate
4. The method changes to ‘finished’ if the user gives a ‘bye’ act.

#### Requested slots
Modern task-based dialog systems are based on a **domain ontology**, a knowledge domain ontology structure representing the kinds of intentions the system can extract from user sentences. The ontology defines one or more frames, each a collection of **slots**, and defines the **values** that each slot can take.  
Each slot is classified as “informable” or not. An “informable” slot is one which the user can provide a value for, to use as a constraint on their search.  Some slots like telephone number are not informable because user may never use them to search anything.(E.g. user may never look up a restaurant by phone number in restaurant recommendation domain.)   
Requested slots are slots user has requested and the system should inform.

### dialog state tracking(DST)
As mentioned before, dialog manager takes input from NLU, and maintain a dialog state, this process could be integrated into a subtask of dialog management, dialog state tracking.   
A *dialog state tracker* takes all of the observable elements up to time t in a dialog as input, including all of the results from the ASR and NLU components, all system actions taken so far, and external knowledge sources such as databases and models of past dialogs.  
Because the ASR and SLU are imperfect and prone to errors, they may output several conflicting, usually an N-Best list of interpretations.  The true state cannot be directly observed. So we need a tracker, given these inputs, to output its estimate of the current dialog state $$s$$ and to correctly identify the true current state $$s^∗$$ of the dialog.  

### DST Methods
The task of DST is to output a distribution over all of the components of the dialog state at each turn. The distributions output by a dialog state tracker are sometimes referred to as the tracker’s *belief* or the *belief state*.

#### Rule-based DST
A common rules-based dialog state tracker used in industry can be described as follows:  
1. keep a single top hypothesis with maximum confidence scores for each component NLU result.  
2. components are shared in the session.  
3. new values override old values for the same component. 
4. complex sharing/overriding rules could be incorporated by system designers

Values of each components of dialogue state come from either current turn's top NLU hypothesis, or previous k turns' states. This simple tracker usually has strong performance because the top NLU result is very likely to be correct because the confidence score has already been trained in slot-specific domain data.   
Rules-based methods requires no model training process, which means no extra task-specific data is needed, and in most real domains, the dst data is very scarce.  The performance of rule-based method is relatively strong because the top NLU result is by far most likely to be correct because the confidence score was already trained in domain-specific data.   
However this method is unable to make use of the entire ASR N-best or NLU M-best lists and do not account for uncertainty(from both errors in recognizing speech, and ambiguities inherent in natural language) in a principled way

#### Generative state tracking
Generative state tracking approaches treat dialogue state $$s$$ and user true action $$u$$ as some hidden variables, and the NLU M-best list are generated from these hidden variables.
<center><img src="/assets/img/dialogue/dynamic_bayesian_dst.jpg" alt="dynamic_bayesian_dst" style="width: 400px;"/></center>

Figure above is a typical generative structure for dst. Two consecutive time steps are shown. A prime $$'$$ denotes a variable in the following time step. The true dialog state $$s$$ is depend on the previous true dialogue state and the machine action $$a$$, the user's action $$u$$ depends on both $$s$$ and $$a$$, the NLU M-best list $$\tilde{u}$$ is a noisy observation of $$u$$. Observable nodes are shaded grey. The connections in the graph encode the conditional dependence relationships in the joint distribution of these random variables. So the dialog state distribution could be estimate from. 

$$
b'(s) = \eta \sum_u{P(\tilde{u}|u)P(u|s, a)b(s)}
$$

where $$\eta$$ is a normalization constant.  
Generative approaches model the posterior over all possible dialog state hypotheses, including those not observed in the NLU M-best list, which is intractable because the whole dialog state space size is to large. One approach to scale up is to group $$s$$ into few partitions, and to track only states suggested by NLU results. Another approach is to factor the components of dialog states, make conditional independence assumptions between the components.  



## Reference
1. Matthew henderson, "Machine Learning for Dialog State Tracking: A Review"
2. Jason D. Williams，"The Dialog State Tracking Challenge Series: A Review"
3. Matthew Henderson, "Dialog State Tracking Challenge 2 & 3"
4. Gasic and Young, "Effective Handling of Dialogue State in the Hidden
Information State POMDP-based Dialogue Manager"
5. Daniel Jurafsky & James H. Martin, "Speech and Language Processing, Chapter 29 Dialog Systems and Chatbots"


