

# Making HED meaningful

This tutorial explains how to create HED annotations that are meaningful, unambiguous, and machine-actionable. Understanding HED annotation semantics is essential for creating annotations that accurately represent what happened during an experiment and can be correctly interpreted by both humans and computers.

A HED annotation consists of tags selected from the HED vocabulary (schema), optionally grouped using parentheses, that together describe events, stimuli, actions, and other aspects of an experiment.

Read the [Introduction to HED](IntroductionToHed.md) for a basic introduction to HED before starting this tutorial.

## Background

### Syntax versus semantics

**HED syntax errors** are structural violations that prevent an annotation from being properly parsed or validated.

**HED semantic errors** refer to annotations that are syntactically correct but fail to accurately or unambiguously convey the intended meaning.

**Example:** Common syntax errors

- Mismatched parentheses: `(Red, Circle))` or `(Red, (Circle)`
- Missing commas between tags: `Red Circle` instead of `Red, Circle`
- Using tags that don't exist in the schema
- Violating tag properties defined in the schema (e.g., extending a tag that doesn't allow extension, omitting required values for value-taking tags)


**HED validators only check for syntax errors.** This document focuses mainly on **semantic errors** - helping you create annotations that are not only syntactically valid but also meaningful, unambiguous, and correctly represent what happened in your experiment.

**HED quality assessment tools are available to assess whether HED annotations are meaningful.** See the [HED validation guide](HedValidationGuide.md) for validation workflows and the [HED summary guide](HedSummaryGuide.md) for summary-based quality checks.

The semantic interpretation of a HED annotation depends on:

1. **Which tags are selected** - Each tag has a specific meaning in the HED vocabulary
2. **How tags are grouped** - Parentheses bind tags that describe the same entity or relationship
3. **Where tags are placed** - Top-level (not inside any parentheses) vs nested (inside parentheses) grouping affects interpretation
4. **The context of use** - Whether the annotation appears in timeline or descriptor data


**Understanding tag placement**

**Top-level tags:** appear outside all parentheses. In `Sensory-event, (Red, Circle)`, the tag `Sensory-event` is top-level.

**Nested tags:** appear inside parentheses. In `Sensory-event, (Red, Circle)`, the tags `Red` and `Circle` are nested within a group.


Tag placement determines scope and relationships - top-level tags typically classify the entire annotation, while nested usually tags describe specific entities or relationships.

### Context for the examples

The main standards for storing imaging and behavioral data in neuroscience are the Brain Imaging Data Structure ([BIDS](https://bids.neuroimaging.io/index.html)) and Neurodata Without Borders ([NWB](https://nwb.org/)). This document uses the BIDS format for its examples, but NWB has equivalent representations. See the [NWB HED extension docs](https://www.hedtags.org/ndx-hed) for examples in NWB.

One approach for annotating tabular data is to use a `HED` column in the table to annotate each row individually. An alternative is to provide dictionaries of annotations that apply to all rows and use tools to "assemble" the actual annotation. Dictionaries save a lot of time and duplication. We assume that such annotation dictionaries go in a JSON sidecar for BIDS (or a `MeaningsTable` for NWB).

The examples assume that you understand the mechanics of assembly of annotations and emphasize what is being assembled rather than how it is assembled. See [Rule 6: Use curly brace assembly](#rule-6-use-curly-brace-assembly) for a brief description of annotation assembly and the [BIDS annotation quickstart](BidsAnnotationQuickstart.md) tutorial for a more detailed explanation of how assembly works in BIDS.

## The reversibility principle

**The reversibility principle**

**A well-formed HED annotation can be translated back into a coherent English description.**



The reversibility principle provides a practical test for whether your HED annotation is semantically correct: Can you translate it back into coherent English?

**Example:** A reversible HED annotation

```
 Sensory-event, Experimental-stimulus, Target, Visual-presentation, 
((Green, Triangle), (Center-of, Computer-screen))
```


**Why this is reversible:**

The sentence can be unambiguously translated as: "A sensory event that is a target experimental stimulus consists of a visual presentation of a green triangle that appears at the center of the computer screen."

- Each group describes a single entity or relationship
- The overall structure tells a coherent story
- `Sensory-event` indicates this is a sensory presentation
- `Experimental-stimulus` indicates this is a task stimulus
- `Target` indicates the task stimulus role
- `Visual-presentation` specifies the sensory modality
- `(Green, Triangle)` - grouped properties describe ONE object
- `(Center-of, Computer-screen)` - spatial relationship (see [Rule 5: Nest binary relationships](#rule-5-nest-binary-relationships) for relationship patterns)
- The outer grouping `((Green, Triangle), (Center-of, Computer-screen))` connects the object to its location

**Example:** A non-reversible HED annotation


```
Green, Red, Square, Triangle, Center-of, Visual-presentation, Sensory-event, Computer-screen
```


**Why this fails reversibility:**

We can determine that this is a sensory event presented visually because of the semantic rules for `Event` tags and `Sensory-presentation` tags as explained in this document. However, the remaining tags: `Green`, `Red`, `Square`, `Triangle`, `Center-of`, and `Computer-screen` cannot be disambiguated:

- Cannot tell if green and red describe the triangle or the square or something else
- Spatial information is disconnected

**A simple reversibility test:** Randomly shuffle the order of the tags and tag groups (keeping the same nesting) and see if you interpret the annotation in the same way.

## Timeline vs descriptor data

The semantic requirements for HED annotations depend on whether they appear in **timeline data** or **descriptor data**.

**HED annotation strategy depends on data type**

**Timeline annotations:** describe what happens at time points during an experiment (e.g.delivery of stimuli or participant responses).

**Descriptor annotations:** provide static descriptions or metadata about entities (e.g., participant characteristics such as age).


[BIDS](https://bids.neuroimaging.io/index.html) stores timeline data in `.tsv` files each with an`onset` column that gives the time in seconds relative to the start of the recording (e.g., `_events.tsv`). BIDS descriptor data is stored in `.tsv` files that do not have an `onset` column (e.g., `participants.tsv`). BIDS associates additional metadata for these files in similarity named JSON (e.g., `_events.json`) files. Information from a `.tsv` file and its associated `.json` files is combined to form assembled HED annotations for the data. The data for an experiment is spread over multiple BIDS `.tsv` and `.json` files.

An [NWB](https://nwb.org/) file is a container that holds all the data for an experiment. The timeline and descriptor data for an experiment are held in `DynamicTable` objects. The `DynamicTable` objects for timeline data have a time-stamp column of some sort.

**Quick rule**

Timeline annotations should include an `Event` tag; descriptor annotations should not.


### Timeline data requires Event tags

Timeline data has timestamps indicating when things happen. Every assembled annotation SHOULD include a single `Event` type tag or multiple `Event` type tags, each in a separate top-level group.

**Example:** Correct timeline annotation (BIDS)


**Excerpt from:** `events.tsv`
| onset | duration | event_type | 
|-------| -------- | ---------- |
| 2.5   | n/a      | square     |

**Sidecar:** `events.json`
```json
{
  "event_type": {
    "HED": {
      "square": "Sensory-event, Experimental-stimulus, Non-target, Visual-presentation, (Red, Square)"
    }
  }
}
```

**Assembled annotation:**
```
Sensory-event, Experimental-stimulus, Non-target, Visual-presentation, (Red, Square)
```


**Why it's correct:**

- Includes `Event` tag: `Sensory-event`
- Includes `Task-event-role`: `Experimental-stimulus`
- It is an experimental stimulus specifying the task stimulus role: `Non-target`
- It is a sensory event specifying the sensory modality: `Visual-presentation`
- Properly groups stimulus properties: `(Red, Square)`

### Descriptor data has no Event tags

Descriptor data (e.g., tables from files such as`participants.tsv` or `samples.tsv`) describes properties or characteristics, not events. `Event` tags SHOULD NOT appear in descriptor data.

**Example:** Correct descriptor data annotation (BIDS)


**Excerpt from:** `participants.tsv`

| participant_id | age | hand  |
|----------------|-----|-------|
| sub-001        | 25  | right |

**Sidecar:** `participants.json`
```json
{
  "age": {
    "HED": "Age/# years"
  },
  "hand": {
    "HED": {
      "right": "Right-handed",
      "left": "Left-handed"
    }
  }
}
```

**Assembled annotation:**
```
Age/25 years, Right-handed
```


**Why it's correct:**

- Describes participant properties: 25 years old and right handed
- No event classification
- No temporal tags: `Onset`/`Offset`
- Semantically appropriate for descriptor context

### Temporal scope tags

Temporal scope tags (`Onset`, `Offset`, `Inset`, and `Delay`) are ONLY for timeline data and indicate the time course of events. `Duration` can be used in either type of data but cannot be used with `Onset`, `Offset`, and `Inset`, which are associated with explicit time point markers in the timeline data. In timeline data `Duration` represents something starting at the current time and extending for a specified amount of time from that point. In descriptor data `Duration` represents a property of something.

See [Temporal annotation strategies](#temporal-annotation-strategies) for detailed guidance on when to use `Duration` versus `Onset`/`Offset` patterns.

## Semantic rules

Rules 2 and 3 in this section primarily apply to **timeline** data (e.g., events with timestamps). For **descriptor** data, event-related rules do not apply. See [Timeline vs descriptor data](#timeline-vs-descriptor-data) above for the distinction.

Remember that HED vocabularies maintain a strict taxonomical or is-a relationship of child tags to parents. When we say `Event` tag, we mean `Event` or any tag that is a descendent of `Event` in the HED vocabulary hierarchy.

**HED annotations are unordered**

**The order of tags in a HED annotation does not affect its meaning.** 


The annotations `Red, Circle` and the annotations `Circle, Red` are semantically equivalent—-both are just a list of two independent tags.

**Parentheses are essential for conveying relationships and ordering**: they explicitly bind tags together to show which tags describe the same entity or relationship.

- Without parentheses: `Red, Circle` is ambiguous (could be two separate things)
- With parentheses: `(Red, Circle)` is unambiguous (one red circle)

Tags within a group are semantically bound and work together to describe one thing. Tags outside the group describe different aspects or entities.

### Rule 1: Group tags for meaning

Grouping rules apply to both timeline and descriptor data and typically refer to tags from the `Item` and `Property` hierarchies.

#### Group object properties

Tags describing properties of the same object MUST be grouped together:

**Example:** Correct object grouping


```
(Red, Circle)
```


The annotation indicates a single object that is both red AND circular. Without grouping the meaning can be ambiguous and could mean two different items, one that is circular and one that is red.

If an item has multiple properties, they should all be grouped together:

**Example:** Grouping of multiple object properties


```
(Green, Triangle, Large)
```


The grouping indicates that this is a large green triangle. Without grouping, these three properties may or may not apply to the same object.

#### Separate unrelated concepts

Tags that describe independent aspects or unrelated concepts SHOULD NOT be grouped together. Don't group tags with no semantic relationship.

**Examples:** Incorrect grouping


- `(Red, Press)` - Color and action are unrelated
- `(Triangle, Mouse-button)` - Stimulus shape and response device are unrelated
- `(Green, Response-time)` - Color and temporal measure are unrelated


### Rule 2: Classify events carefully

Event and other timeline data is usually stored in a tabular format where each row marks a point on the experimental timeline and represents one or more "events" (something that happens).

**Core requirements for annotating an event**

  - Every event MUST have exactly one tag from the `Event` hierarchy
  - If there is a task, an event annotation SHOULD have a `Task-event-role` tag
  - A sensory event SHOULD have a `Sensory-presentation` tag
  - Each event annotation should be in a separate group if multiple events occur at the same time


#### 2a. Event tags categorize

The `Event` tags provide the primary classification or anchors for event annotations.

| Event                  | Description                                                                                          | Implicit agent             |
| ---------------------- | ---------------------------------------------------------------------------------------------------- | -------------------------- |
| `Sensory-event`        | A sensory presentation occurs                                                                        | One experiment participant |
| `Agent-action`         | An agent performs an action                                                                          | One experiment participant |
| `Data-feature`         | A computed or observed feature is marked                                                             | Software agent             |
| `Experiment-control`   | Experiment structure or parameters change (e.g., stop signal presented sooner in GO-NOGO experiment) | Controller agent           |
| `Experiment-procedure` | Experiment paused to administer something (e.g., a quiz or saliva test)                              | Experimenter               |
| `Experiment-structure` | Organizational boundary or marker (e.g., start of a trial or block)                                  | Controller agent           |
| `Measurement-event`    | A measurement is taken (which may be recorded elsewhere)                                             | Controller agent           |

**Example: Single sensory event with proper classification**

```
Sensory-event, Experimental-stimulus, Visual-presentation, (Red, Circle), (Green, Square)
```


The event is a sensory event (from the perspective of the experiment participant) which is an experimental stimulus consisting of the simultaneous presentation of a red circle and a green square.

**Classification breakdown:**

- `Sensory-event` - Event type (from `Event` hierarchy)
- `Experimental-stimulus` - Task role (from `Task-event-role` hierarchy)
- `Visual-presentation` - Sensory modality (from `Sensory-presentation` hierarchy)
- `(Red, Circle), (Green, Square)` - Describes what is presented to the senses

A single top-level `Event` tag is assumed to represent an event that includes all of the rest of the tags in the annotation. The sensory event in the example is an experimental stimulus (something that the participant will need to act on as part of the experiment's task). This is the most common method of annotating events.

#### 2b. Task event role qualifiers

If an experiment involves a task, each event should be associated with a `Task-event-role`:

- `Experimental-stimulus` - Primary stimulus participant must detect, identify, or respond to
- `Cue` - Signal indicating what to expect or do next
- `Participant-response` - Action by the participant
- `Feedback` - Information about participant's performance
- `Instructional` - Task instructions or information
- `Warning` - Alert or warning signal
- `Incidental` - Present but not task-relevant
- `Task-activity` - Marker of ongoing task activity period
- `Mishap` - Unplanned occurrence affecting experiment

#### 2c. Task-stimulus-role qualifiers

If the event task role is `Experimental-stimulus`, tags from the `Task-stimulus-role` hierarchy provide important information about the task stimulus. For example, tags such as `Penalty` or `Reward` are often used to modify the `Feedback` role. If the annotation contains an `Experimental-stimulus` tag, consider whether any tags from `Task-stimulus-role` are appropriate. Common qualifiers include:

- `Target` - The thing the participant should focus on or respond to
- `Non-target` - Something to ignore or not respond to
- `Expected` - Stimulus matches what was cued
- `Unexpected` - Stimulus differs from what was cued
- `Penalty` - Negative consequence for performance
- `Reward` - Positive consequence for performance

**Example:** Stimulus with task role qualifier


```
Sensory-event, Experimental-stimulus, Target, Visual-presentation, (Red, Circle)
```


Th annotation indicates a visual experimental stimulus target - a red circle that the participant should be specifically looking for.

#### 2d. Task-action-type qualifiers

Tags from the `Task-action-type` hierarchy provide important information about the nature of the participant's response. If the annotation contains a `Participant-response` tag, consider whether any tags from `Task-action-type` are appropriate. Common qualifiers include:

- `Correct-action` - Response matches task requirements
- `Incorrect-action` - Response does not match task requirements
- `Appropriate-action` - Action is suitable in context
- `Inappropriate-action` - Action is unsuitable in context
- `Switch-attention` - Participant shifts focus
- `Near-miss` - Almost correct response

**Example:** Response with action qualifier


```
Agent-action, Participant-response, Correct-action, (Experiment-participant, (Press, Mouse-button))
```


The annotation indicates that the experiment participant pressed the mouse button, and this was a correct response to the task.

#### 2e. Sensory presentations

If the event is a `Sensory-event`, a `Sensory-presentation` tag (e.g., `Visual-presentation` or `Auditory-presentation`) SHOULD be included to specify what senses are affected by the presentation. This is essential for search and query functionality.

#### 2f. Handling multiple events

If a single row annotation contains multiple events, the tags relevant to each event must be separately grouped in parentheses.

**Example:** A row annotation represents multiple sensory events


**Excerpt from:** `events.tsv`

| onset | duration | visual_type   | auditory-type |
| ----- | -------- | ------------- | ------------- |
| 104.5 |  'n/a'   |  show_circle  | sound_green   |

**Asssembled annotation:**
```
(Sensory-event, Experimental-stimulus, Visual-presentation, (Red, Circle)), (Sensory-event, Experimental-stimulus, Auditory-presentation, (Word, Label/Green))
```


The annotation (from the perspective of the experiment participant) consists of two simultaneous sensory events -- a red circle (usually assumed to be displayed on the computer screen if no other information is present) and a spoken word "Green". This type of annotation often occurs in congruence experiments or attention shifting experiments.

It is also possible to annotate this as a single sensory event that is an experimental stimulus with two modalities of presentation. The choice should be made consistently, but if the two presentations have different task roles or are expected to elicit separate cognitive responses, they should always be annotated separately as in the example.

**Example:** Multiple rows have the same time.


**Excerpt from:** `events.tsv`

| onset | duration | event_type      |
| ----- | -------- | --------------- |
| 104.5 |  'n/a'   |  show_circle    | 
| 104.5 |  'n/a'   |  sound_green    |

**Asssembled annotation:**
```
(Sensory-event, Experimental-stimulus, Visual-presentation, (Red, Circle)), 
(Sensory-event, Experimental-stimulus, Auditory-presentation, (Word, Label/Green))
```


The meaning of this annotation is the same as in the previous example where the annotations are in one row. They are distinct sensory events and their respective tags must be grouped separately regardless of where they appear.

**Note:** The annotations for rows with the same times (regardless of where the rows appear in the table) are concatenated to form a single annotation. The assembled annotation cannot have duplicates (either tags or groups) regardless of whether the duplicates were are different rows, if the markers have the same time.

Another common situation is data in which the response time to an event is in the same row as the stimulus presentation. Use the `Delay` tag to convey the timing as illustrated in the following example:

**Example:** An annotation for row with a stimulus and response time.


**Excerpt from:** `events.tsv`
| onset | duration | stimulus | responseTime |
| ----- | -------- | -------- | ------------ |
| 104.5 |  'n/a'   |  circle  |   0.250      |

**Asssembled annotation:**
```
(Sensory-event, Experimental-stimulus, Visual-presentation, Circle),
(Delay/0.250 s, (Agent-action, Participant-response, (Experiment-participant, (Push, Mouse-button))))
```


At time 104.5 seconds into the experiment a circle is presented on the computer screen, and the participant takes 0.250 seconds to push a mouse button in response to the presentation. This annotation represents two separate events:

- An experimental stimulus that is the visual presentation of a circle (assumed to be on the screen) at time 104.5 seconds from the start of the experiment.
- A participant response consisting of the experiment participant pushing the mouse button at 104.750 seconds from the start of the experiment.

### Rule 3: Understand perspective


**Key principle**

**Every type of event has a perspective that informs the viewpoint of the annotation.** 


Perspective is generally a property of timeline data not descriptor data. Correct identification of the perspective allows downstream tools to assess the influence of the event on the participants' cognition and behavior. Event annotations that contain `Agent` and/or `Agent-task-role` tags have **explicit** perspective, while those without those tags have **implicit** perspective. See the Event table in [Rule 2: Classify events carefully](#rule-2-classify-events-carefully) for the implicit agent associated with each event type.

#### Sensory event perspective

Sensory events are assumed to be from the perspective of the single experiment participant unless explicitly tagged to the contrary.

**Example:** Participant perspective for sensory event (implicit)


```
Sensory-event, Cue, Visual-presentation, (Red, Circle)
```


In this sensory event, the participant sees a red circle on screen meant to be a cue to the participant to get ready to respond. The agent is assumed to be a human agent whose role is as the single experiment participant. The perspective is implicit because the agent and the agent's role in the task are not explicitly tagged for this event.

**Why it works:** Usually sensory events do not have `Agent` and `Agent-task-role`, and the annotation is assumed to describe the experiment from the viewpoint of a single human participant.

#### Agent action perspective

For `Agent-action` events, the actor performing the action can be specified with varying levels of detail:

**Agent TYPE vs Agent ROLE:**

- **Agent type** (from `Agent` hierarchy): `Human-agent`, `Animal-agent`, `Avatar-agent`, `Controller-agent`, `Robot-agent`, `Software-agent`
- **Agent role** (from `Agent-task-role` hierarchy):  `Experiment-actor`, `Experiment-controller`, `Experiment-participant`, `Experimenter`


##### Implicit agents

If an `Agent-action` appears without explicit agent or agent task role tags, a single experiment participant is assumed by default. The characteristics of the agent as defined by the `Agent` tag (e.g., `Human-agent` or `Animial-agent`) may be specified or assumed to be provided by additional data, such as the `participants.tsv` file in BIDS.

**Example:** An implicit agent is assumed

```
Agent-action, Participant-response, (Press, Mouse-button)
```


The annotation indicates that the single human experiment participant presses the mouse button.

##### Agent role requirements

Use `Experiment-participant`, `Experimenter`, or other `Agent-task-role` tags when:

- Multiple experiment participants are involved
- Agents are not the experiment participant
- When the experiment participant is not human
- Clarity about who did what is important
- You want to be explicit for consistency

**Example:** An experiment with two partipants playing Jeopardy

```
Agent-action, Participant-response, ((Experiment-participant, ID/sub_003), (Press, Mouse-button))
```


In this experiment either participant could have pressed their mouse button and so their responses must be distinguished in the annotation.

##### Agent type requirements

Use `Animal-agent`, `Robot-agent`, or other `Agent` tags when the agent is NOT a human:

**Example:** A mouse presses a lever for a reward

```
Agent-action, Participant-response, ((Animal-agent, Animal/Mouse), (Press, Lever))
```


The annotation indicates the participant, a mouse, presses a lever. The `Experiment-participant` is implicit in this annotation, but could be made explicit by using `(Animal-agent, Experiment-participant, Animal/Mouse)` in the example.

Note that since `Mouse` is not a tag in the schema, it must be modified by its closest potential parent in the schema: `Animal/Mouse`. (See [Rule 8: Extend tags carefully](#rule-8-extend-tags-carefully) for guidance on extending tags.)

**Example:** An avatar in a virtual reality experiment interacts with a human

```
Agent-action, ((Avatar-agent, Experiment-actor, ID/34A7), (Collide-with, Building))
```


The avatar is not labeled with `Experiment-participant` but with `Experiment-actor`. It is part of the scenario, but we are not measuring its cognition or behavior.

**Best practices:**

- In human experiments: `Human-agent` can be omitted (it's implicit)
- In animal/robot experiments: Usually specify the agent type (`Animal-agent`, `Robotic-agent`)
- Be consistent throughout your dataset

See [Rule 4](#rule-4-nest-agent-action-object) for the complete agent-action-object structural pattern.

More complicated scenarios (e.g., multiple participants, agents that are not human, or agents that are not the experiment participant) are also possible to annotate unambiguously, but in these cases the `Agent` and/or `Agent-task-role` are required for unambiguous annotation.

### Rule 4: Nest agent-action-object

Agent-action-object relationships require nested grouping to show who did what to what.

**Pattern: Nesting structure for agent-action-object**

```
(Agent-tag, (Action-tag, Object-tag))
```


The grouping is is meant to convey normal sentence structure: subject predicate direct-object. This annotation indicates that the agent performs the action on the object.

**Example:** The human experiment participant correctly presses the left mouse button in response to the task.

```
Agent-action, Participant-response, Correct-action, (Experiment-participant, (Press, (Left, Mouse-button)))
```


TThis example shows minimal grouping -- there could be additional grouping for clarity, but this minimal grouping should be unambiguous.

**Structure Explanation:**

- `Agent-action` - `Event` top-level classification
- `Participant-response` - `Task-event-role` modifier
- `Correct-action` - `Task-action-type` explains what the action means for the task
- Outer action group: `(Experiment-participant, (Press, (Left, Mouse-button)))` connects agent to action
- Inner Tag (or group): `Experiment-participant` - describes WHO does the action
- Inner group with an `Action` tag: `(Press, (Left, Mouse-button))` - describes WHAT action on WHICH object

If a tag from the `Action` hierarchy is ungrouped, it cannot be determined syntactically who is the actor (`Experiment-participant` or `Mouse-button`).

**Example:** Incorrect agent-action structure


```
Agent-action, Experiment-participant, Press, Mouse-button
```


Without grouping indicates WHO did WHAT. The relationships are lost, making the annotation semantically incomplete. This annotation only indicates that the an experiment participant exists but does not capture the directional relationship. Did the mouse-button press the participant or vice versa?

### Rule 5: Nest binary relationships

Most tags from the `Relation` tag hierarchy express directional binary relationships and REQUIRE specific nested grouping to disambiguate. The only exceptions are the logical relation tags `AND` and `OR`, which allow the combination of multiple binary relationships acting on the same source and target.

**Relation tag syntax**

```
(A, (Relation, C))
```


The annotation specifically designates a direction "A → C" through the binary `Relation` tag. In interpreting relation groups:

- **A** is the source/subject of the relationship
- **Relation** is the binary directional relationship (from `Relation` hierarchy)
- **C** is the target/object of the relationship
- The relationship flows from **A** to **C** through the `Relation` tag

Relationship grouping has the following structure:

- Outer parentheses group the entire relationship
- Inner parentheses group the relation with its target
- The source appears in the outer group

**Example:** Spatial relationship pattern

```

((Red, Circle), (To-left-of, (Green, Square)))


This annotation indicates a red circle is to the left of a green square.

**Example:** A size comparison

```
Sensory-event, Experimental-stimulus, Visual-presentation, ((Cross, White, Size), (Greater-than, (Circle, Red, Size)))
```


This annotation indicates an experimental stimulus consists of a white cross and a red circle. The white cross is bigger than the red circle.

**Example:** Using AND to combine operations

```
(Cross, ((Close-to, AND To-left-of), Square))
```


This annotation indicates a cross that is close to and to the left of a square. The `AND` and `OR` relation tags should only be used when the source and target are the same.

Common `Relation` tags include:

**Spatial relations:**

- `To-left-of`, `To-right-of` - horizontal positioning
- `Above`, `Below` - vertical positioning
- `Center-of`, `Edge-of`, `Corner-of` - reference positioning
- `Near`, `Far-from` - distance relations

**Temporal relations:**

- `Before`, `After` - sequential ordering
- `During` - containment in time
- `Synchronous-with` - simultaneous occurrence

**Hierarchical relations:**

- `Part-of` - component relationship
- `Member-of` - membership relationship
- `Contained-in` - inclusion relationship

**Important:** The order matters! `(A, (To-left-of, B))` means "A is to the left of B", which is different from `(B, (To-left-of, A))` which means "B is to the left of A".

### Rule 6: Use curly brace assembly

**Assembly** refers to the process of looking up the applicable annotations for each row of a table and creating a complete HED annotation for that row. HED concatenates the annotations associated with each column by default. This works for independent information but fails when multiple columns describe parts of the same entity and the result does not use parentheses properly.

An alternative is to create an assembly template using the curly brace syntax.

**Example:** Ambiguous annotation with flat concatenation (BIDS)


**Excerpt from:** `events.tsv`
| onset | duration | event_type | color | shape  |
| ----- | -------- | ---------- | ----- | ------ |
| 4.8   | n/a      | visual     | red   | circle |

**Sidecar:** `events.json`
```json
{
  "event_type": {
    "HED": {
      "visual": "Sensory-event, Experimental-stimulus, Visual-presentation"
    }
  },
  "color": {
    "HED": {
      "red": "Red"
    }
  },
  "shape": {
    "HED": {
      "circle": "Circle"
    }
  }
}
```

**Assembled annotation:**
```
Sensory-event, Experimental-stimulus, Visual-presentation, Red, Circle
```


**Problem:** The `Red` and `Circle` are separate top-level tags so we cannot definitively determine whether they describe the same object. We can solve this problem by using curly braces in the annotation dictionary (JSON sidecar for BIDS) to specify how the annotations for the individual columns should be assembled.

**Example:** Using a curly brace template to disambiguate (BIDS)


**Sidecar:** `events.json`
```json
{
  "event_type": {
    "HED": {
      "visual": "Sensory-event, Experimental-stimulus, Visual-presentation, ({color}, {shape})"
    }
  },
  "color": {
    "HED": {
      "red": "Red"
    }
  },
  "shape": {
    "HED": {
      "circle": "Circle"
    }
  }
}
```

**Assembled annotation:**
```
Sensory-event, Experimental-stimulus, Visual-presentation, (Red, Circle)
```


**Why it works:** The curly braces `{color}` and `{shape}` contain column names that are replaced by their HED annotations when the annotation is assembled. This placement inside of the parentheses of the anchor annotation (the `event_type` column) assures they are grouped as properties of the same object. Without curly braces, annotations for each column in a table row are simply concatenated (joined with commas) to form an assembled annotation for the row.

**Curly braces control how an annotation is assembled**

**Use curly braces when:**

- Multiple columns contribute properties of the SAME entity (e.g., color + shape = one object)
- You need to control grouping across columns in sidecars
- Flat concatenation would create ambiguous relationships

**Don't use curly braces when:**

- Each column describes independent aspects (naturally separate)
- Annotating directly in a HED column (not using a sidecar)
- All tags naturally group correctly without templates


The alternative to using sidecars for annotations is to create a HED column in the table. However this requires an individual annotation for each row, while the sidecar approach allows reuse of annotations across many rows.

### Rule 7: Heed special HED syntax

HED has some special syntax rules which are encoded in the HED schema as schema attributes and properties.

#### Reserved tags

Reserved tags (tags with the `reserved` schema attribute) implement HED infrastructure. These tags have special grouping rules and usage patterns as shown in the following table:

| Tag | Example | Rules |
| --- | --- | --- |
| `Definition` | `(Definition/Red-triangle, (Sensory-event, Visual-presentation, (Red, Triangle)))` | * Used to name frequently used strings * Can only be defined in sidecars or externally * Defining tags are in inner group * Definition group cannot contain any other reserved tags |
| `Def` | `Def/Red-triangle` | * Used to anchor `Onset`, `Inset`, `Offset` * Must correspond to a `Definition` * Cannot appear in definitions |
| `Def-expand` | `(Def-expand/Red-triangle, (Red, Triangle))` | * Used to anchor `Onset`, `Inset`, `Offset` * Must correspond to `Definition` * Cannot appear in definitions * DO NOT USE -- Tools use during processing |
| `Onset` | `(Def/Red-triangle, Onset)` | * Mark the start of an unfolding event * Must have a `Def` anchor corresponding to a `Definition`. * Can include an internal tag group * Must be in a top-level tag group * Event ends when an `Offset` or `Onset` with same `Def` name is encountered |
| `Offset` | `(Def/Red-triangle, Offset)` | * Marks the end of an unfolding event * Must have a `Def` tag anchor * Must be in a top-level tag group * Must correspond to an ongoing `Onset` group of same name |
| `Inset` | `(Def/Red-triangle, (Luminance-contrast/0.5), Inset)` | * Marks an interesting point during the unfolding of an event process. * Must have a `Def` tag anchor * Can include an internal tag group * Must be in a top-level tag group * Must be between an `Onset` and the ending time of that event process |
| `Duration` | `(Duration/2 s, (Sensory-event, Auditory-presentation, Feedback Buzz))` | * Designates length of event that starts at that point * Must be in a top-level tag group * Cannot be used in group with `Onset`, `Inset`, or `Offset`. * May be used in both timeline and description data. * Inner tag group defines the event that starts at that point * If used with `Delay`, the event start is delayed by indicated amount |
| `Delay` | `(Delay/0.5 ms, (Sensory-event, Auditory-presentation, Feedback Buzz))` | * Delays the start of the event by the indicated amount * Must be a top-level tag group |
| `Event-context` | `(Event-context, (Def/Task-a, (Blue, Square)), (Def/Task-b, (Red, Triangle)))` | * Should only be created by tools * Must be a top-level tag group * Keeps track of ongoing events at intermediate time points * Represents concurrent active event processes |


#### Values and units

HED **value-taking tags** (also called placeholder tags), are indicated in the schema by a `#` symbol in the schema.

Some of these tags have attributes specifying the types of values that These tags require specific values to complete them, and the schema defines the types of values and units are allowed.

**Value classes** define what type of value can be used:

- `nameClass` - Alphanumeric names (letters, digits, hyphens, underscores): `Label/My-stimulus-1`
- `textClass` - Any printable UTF-8 characters: `ID/Participant answered: "Yes!"`
- `numericClass` - Valid numeric values: `Duration/2.5 s`, `Age/25 years`

**Unit classes** define physical quantities that have units of measurement:

- `timeClass` - Time measurements: `Duration/500 ms`, `Delay/2.3 s`
- `physicalLengthClass` - Spatial measurements: `Distance/50 cm`, `Height/1.8 m`
- `angleClass` - Angular measurements: `Angle/45 degrees`
- `frequencyClass` - Frequency measurements: `Frequency/60 Hz`
- `speedClass` - Velocity measurements: `Speed/25 m-per-s`
- `intensityClass` - Intensity measurements: `Sound-volume/80 dB`

**Example:** Using value-taking tags correctly

```
(Duration/1.5 s, (Sensory-eventLuminance/100 cd-per-m^2, Label/Fixation-cross))
```


**Key rules for value-taking tags:**

1. Use the `#` placeholder in templates: `Duration/# s` becomes `Duration/2.5 s` on assembly
2. Include units if available in the annotation, not with the value: `Distance/# cm` NOT `Distance/#` with the units in the table as `50 cm`
3. Choose appropriate units from the allowed list in the schema
4. Follow value class restrictions (alphanumeric for `nameClass`, etc.)

The HED schema specifies allowed units for each unit class. For example, `timeClass` allows units like `s` (seconds), `ms` (milliseconds), `minute`, `hour`. Always use the standard unit abbreviations defined in the schema. SI units can use all of the allowed SI unit modifiers.

#### Label, ID, Parameter-name

The `Label` tag provides an identifying name or label for something. Labels have the `nameClass` attribute, meaning that their values must contain only alphanumeric, hyphen, and underbar characters.

**For identifiers that contain arbitrary printing UTF-8 characters**, use the `ID` or `Parameter-name` tags. These two tags have the `textClass` attribute and can take very general values.

A common use these tags is to group a value with tag that does not take a value (i.e., does not have a `#` child) as illustrated in the following example:

**Example:** The word Spanish word teléfono is displayed on the screen.

```
Sensory-event, Experimental-stimulus, Visual-presentation, (Word, ID/teléfono)
```


The annotation is for an experiment in a language experiment, where Spanish words are displayed on the computer screen. We want to give the value of the word being displayed, but `Word` does not have a placeholder child, so we cannot write, `Word/teléfono`. Instead we modify `Word` with the ID of the word. We did not use `Label` because the value requires UTF-8 to properly display the accent.

Another common use is to add non-standard units or other modifiers to tags that take values.

**Example:** Annotate a column containing  the rate of change of temperature

```
(Temporal-rate/#, Label/Degrees-per-second)
```


This might be an annotation for a column table that has values in degrees/second. HED does not have a unit class corresponding to rates for temperatures, so we modify the general `Temporal-rate` tag with the units. We can use `Label` tag for the units because the units only have ASCII characters and hyphens. We could also have used `ID` or `Parameter-value`.

### Rule 8: Extend tags carefully

The HED schema vocabulary hierarchy can be extended to accommodate more specialized annotations. HED library schemas are formal extensions of HED for specialized vocabularies. Users can also extend the hierarchy by appending new tag to an existing tag that allows extension (e.g., `Parent-tag/Tag-extension`).

Tags that can be extended have (or have inherited) the `extensionAllowed` attribute. Tags that can be extended include all tags EXCEPT those in the `Event` or `Agent` subtrees or those with a `#` child (value-taking nodes).

**You should ONLY consider extending the hierarchy if it is necessary to correctly capture the meaning in the context of the annotation.** HED validators will always give a warning if you extend, but valid extensions are not considered to be errors.

When you need to use a tag that doesn't exist in the schema, **EXTEND FROM THE MOST SPECIFIC** applicable parent tag while preserving the is-a (taxonomic) relationship.

**Guidelines for extending HED**

- **Extend from the closest applicable parent** - Follow the schema hierarchy as deep as possible
- **Preserve the is-a relationship** - Your extension should be a specific type of its parent
- **Include only the tag's direct parent** - Don't include the full path -- just immediate parent
- **Check for extension restrictions** - Some tags cannot be extended (`Event` subtree, `Agent` subtree, or # value nodes)


While you must include the extension's immediate schema parent in the annotation, you don't need to include the full path.

Suppose you want to annotate a house, but house is not a tag in the schema.

**Example:** Extending Item hierarchy at the wrong level


**Syntactically correct, but not useful:**
```
Item/House
```

**Correctly using the most specific applicable parent (Building):**

```
Item/Object/Man-made-object/Building/House
```

**Better using short form:**

```
Building/House
```



The difficulty with first annotation is that searches for the `Building` tag will not return annotations containing `House`. `Item` has many levels of hierarchy that better classify house.

Value-taking tags (have `#` children in the schema) also cannot be extended. These tags reside in the `Property` hierarchy in the standard schema. If you really need a new type of acceleration, you can use the grouping strategy illustrated in the following example.

**Example:** Creating a custom acceleration type


**Wrong:**
```
Acceleration/#/Custom-acceleration
```

**Correct:**
```
(Rate-of-change/Custom-acceleration, Parameter-value/#)
```


The `Acceleration` tag has a `#` child, so you cannot extend with `Custom-acceleration`. Instead, you can extend from `Acceleration` tag parent `Rate-of-change` and group with the general-purpose `Parameter-value` tag which takes the value.

Tags in the `Event` and `Agent` hierarchies cannot be extended, but you can qualify them with labels demonstrated for `Acceleration` in the previous example. However, you should avoid this if at all possible. `Event` tags are the primary HED mechanism for event classification and extraction.

## Temporal annotation strategies

When events have duration or unfold over time, you can choose between two annotation strategies: using `Duration` for simple cases or using `Onset`/`Offset` for more complex temporal patterns. Both approaches are valid; choose based on your data structure and analysis needs.

### Strategy 1: Duration

Use `Duration` when you know the event's duration and want to capture it in a single annotation.

**Scenario:** A fixation cross appears and stays on screen for 1.5 seconds starting at 0.5 s from the start of the recording.

**Example:** A fixation cross 3 cm in height appears for 1.5 s starting at 0.5 s (BIDS)



**Excerpt from:** `events.tsv`

| onset | duration | event_type | cross_size |
|-------|----------|------------| ---------- |
| 0.5   | 1.5      | fixation   | 3          |

**Sidecar:** `events.json`
```json
{ 
  "duration": {
    "HED": "(Duration/# s, ({event_type}, {cross_size}))"
  },
  "event_type": {
    "HED": {
      "fixation": "Sensory-event, Visual-presentation, Cue"
    }
  },
  "cross_size": {
    "HED": "(Cross, Height/# cm)"
  }
}
```

**Assembled annotation**:
```
(Duration/1.5 s, (Sensory-event, Visual-presentation, Cue, (Cross, Height/3 cm)))
```


The annotation is for a single time marker and assumes that the duration of 1.5 s is known at the time of onset.

**Why use `Duration`:**

- Captures the information of an ongoing event in a single annotation
- `Duration` is often of direct interest
- Don't need any `Definition` anchors
- Simpler when the event has a known duration at the start of the event

### Strategy 2: Onset/Offset

Use `Onset` and `Offset` when you have separate time markers for the start and end of an event or when you need to mark intermediate time points.

**Example:** Encoding ongoing events using Onset and Offset (BIDS)


**Excerpt from:** `events.tsv`

| onset | duration | event_type     | cross_size |
|-------|----------|----------------| ---------- |
| 0.5   | 'n/a'    | fixation_start | 3          |
| 2.0   | 'n/a'    | fixation_end   | 'n/a'      |


**Sidecar:** `events.json`

```json
{
  "event_type": {
    "HED": {
      "fixation_start": "(Def/Fixation-point, (Sensory-event, Visual-presentation, Cue, {cross_size}), Onset)",
      "fixation_end": "(Def/Fixation-point, Offset)"
    }
  },
  "cross_size": {
    "HED": "(Cross, Height/# cm)"
  },
  "definitions": {
    "HED": {
      "fix_def":  "(Definition/Fixation-point)"
	}
  }
}
```

**Assembled annotation (at 0.5 s):**
```
(Def/Fixation-point, (Sensory-event, Visual-presentation, Cue, (Cross, Height/3 cm)), Onset)
```

**Assembled annotation (at 2.5 s):**
```
(Def/Fixation-point, Offset)
```


These annotations indicate that a fixation cross 3 cm in height starts showing at 0.5s into the recording and disppears at 2.0 s. The anchor `Def/Fixation-point` connects the onset marker (at time 0.5s) and offset marker (at 2.0s) for this display. The grouped content under `Onset` continues for the duration of the event.

**Why use `Onset`/`Offset`:**

- Temporal scope is explicit (the end of the event is an event marker)
- Can explicitly express complicated interleaving of events
- Can use the anchor definitions content to shorten annotations
- Can use the `Inset` mechanism to mark intermediate time features associated of the event
- Better for events with variable or unpredictable durations

Notice that the `Fixation-point` definition doesn't have any content in this example. We didn't put the `Sensory-event` and related tags in the definition because we wanted to get the correct grouping with parentheses.

### Use Delay for response timing

The `Delay` tag is used to indicate that the event starts at a specified delay from the time of the event marker in which the annotation appears. This mechanism is often used when the information about an entire trial (i.e., both stimulus and response) are associated with a single time marker. In this case the annotation may contain multiple events -- some of which are delayed from the event marker time.

**Example:** Each row represents a trial in which a participant response to a circle or square (BIDS)



**Excerpt from:** `events.tsv`

| onset | duration | event_type | response_time | response_type |
|-------|----------|------------| ------------- | ------------- |
| 0.5   |  n/a     | square      | 200.5        | correct       |

**Sidecar:** `events.json`
```json
{ 
  "event_type": {
    "HED": {
      "square": "(Sensory-event, experimental-stimulus, Visual-presentation, Square)"
    }
  },
  "response_time": {
    "HED": "(Delay/# ms, (Agent-action, Participant-response, {response-type}, (Press, (Left, Mouse-button))))"
  }
  "response_type": {
    "HED": {
      "correct": "Correct"
    }
  }
}
```

**Assembled annotation**:
```
 (Sensory-event, Experimental-stimulus, Visual-presentation, Square),
 (Delay/3.5 ms, (Agent-action, Participant-response, {response-type}, (Press, (Left, Mouse-button))))"
```


Two events the stimulus and the response are encoded in a single row. Property tagging with \`Delay encodes them as separate events. Tools can convert this row (the stimulus delivery at 0.5s and the participant's response at 0.7005s) to separate rows if desired.

## Common annotation mistakes

Before finalizing your annotations, review these common mistakes and how to fix them:

**Mistake 1:** Forgetting to group object properties (Violates [Rule 1](#rule-1-group-tags-for-meaning).)

**Wrong:**
```
Sensory-event, Visual-presentation, Red, Circle
```

**Correct:**
```
Sensory-event, Visual-presentation, (Red, Circle)
```


**Mistake 2:** Forgetting the Event tag in timeline data - [Rule 2a](#2a-event-tags-categorize)

**Wrong:**
```
Visual-presentation, (Red, Circle)
```

**Correct:**
```
Sensory-event, Visual-presentation, (Red, Circle)
```


**Mistake 3:** Missing Task-event-role for task events - [Rule 2b](#2b-task-event-role-qualifiers)

**Wrong:**
```
Sensory-event, Visual-presentation, (Red, Circle)
```


**Correct:**
```
Sensory-event, Experimental-stimulus, Visual-presentation, (Red, Circle)
```


**Mistake 4:** Incorrect agent-action structure - [Rule 4](#rule-4-nest-agent-action-object)

**Wrong:**
```
Agent-action, Experiment-participant, Press, Mouse-button
```

**Correct:**
```
Agent-action, (Experiment-participant, (Press, Mouse-button))
```


**Mistake 5:** Incorrect relationship structure - [Rule 5: Nest binary relationships](#rule-5-nest-binary-relationships)

**Wrong:**
```
(Red, Circle, To-left-of, Green, Square)
```

**Correct:**
```
((Red, Circle), (To-left-of, (Green, Square)))
```
**Meaning:** Red circle is to-left-of green square.


**Mistake 6:** Grouping unrelated concepts - [Rule 1](#rule-1-group-tags-for-meaning)

**Wrong:**
```
(Red, Press, Circle)
```

**Correct:**
```
(Sensory-event, Visual-presentation, (Red, Circle)), 
(Agent-action, Participant-response, (Experiment-participant, Press))
```


**Mistake 7:** Using Event tags for descriptor data - [Descriptor data](#descriptor-data-has-no-event-tags)

**Wrong (in participants.tsv):**
```
Sensory-event, Age/25 years
```

**Correct:**
```
Age/25 years
```


**Mistake 8:** Extending from overly-general tags - [Rule 8](#rule-8-extend-tags-carefully)

**Wrong:**
```
Item/House
```

**Correct:**
```
Building/House
```


Extend from the most specific applicable parent. Use short-form for your tags.

**Mistake 9:** Forgetting curly braces for multi-column assembly - [Rule 6](#rule-6-use-curly-brace-assembly)

**Wrong sidecar:**
```json
{
  "event_type": {
    "HED": {"visual": "Sensory-event, Visual-presentation"}
  },
  "color": {"HED": {"red": "Red"}},
  "shape": {"HED": {"circle": "Circle"}}
}
```

**Assembled annotation:**
```
Sensory-event, Visual-presentation, Red, Circle
```

**Correct sidecar:**
```json
{
  "event_type": {
    "HED": {"visual": "Sensory-event, Visual-presentation, ({color}, {shape})"}
  },
  "color": {"HED": {"red": "Red"}},
  "shape": {"HED": {"circle": "Circle"}}
}
```

**Assembled annotation:**
```
Sensory-event, Visual-presentation, (Red, Circle)
```


## Best practices checklist

Use this checklist before finalizing your annotations:

**Checklist:** Creating semantically correct HED annotations

**✓ Grouping**
- [ ] Stimulus properties grouped: `(Red, Circle)` not `Red, Circle`
- [ ] `Agent-action` uses nested structure: `((agent), (action, object))`
- [ ] `Event` tag NOT inside property groups (keep at top level)
- [ ] Unrelated concepts NOT grouped together

**✓ Event Classification**
- [ ] Every timeline event has `Event` tag
- [ ] Every timeline event has `Task-event-role` tag (when applicable)
- [ ] `Sensory-event` includes `Sensory-presentation` tag

**✓ Data Type**
- [ ] Timeline data: `Event` tag present
- [ ] Descriptor data: NO `Event` tags
- [ ] Timeline data only: `Onset`/`Offset`/Inset if needed
- [ ] Descriptor data: NO temporal tags (`Duration` is allowed but interpreted as a description)

**✓ Assembly**
- [ ] Curly braces used for complex grouping (in sidecar)
- [ ] `#` placeholder for numeric values -- units allowed if `#` tag has a `unitClass`
- [ ] Column references match actual column names

**✓ Relationships**
- [ ] Directional relations use `(A, (Relation, C))` pattern
- [ ] Spatial relationships clearly indicate source and target
- [ ] Agent-action-object relationships properly nested

**✓ Definitions**
- [ ] Repeated patterns defined once with Definition/DefName
- [ ] Each Definition name is unique
- [ ] Def/DefName used to reference definitions in annotations
- [ ] Definitions defined in sidecars or externally

**✓ Validation**
- [ ] All tags exist in HED schema
- [ ] Required children specified
- [ ] Extensions have parent tag in the HED schema
- [ ] Units provided where needed and allowed

**✓ Semantics**
- [ ] Annotation translates to coherent English (reversibility test)
- [ ] No ambiguity in interpretation
- [ ] Makes sense in context
- [ ] Consistent structure across similar events

**✓ Style**
- [ ] Multi-word tags use hyphen to separate
- [ ] Consistent capitalization throughout (leading word capitalized)
- [ ] Standard spacing (space after comma)
- [ ] No extra spaces inside parentheses


## Summary

Creating semantically correct HED annotations requires understanding:

1. **The reversibility principle** - Your annotations should translate back to coherent English
2. **Semantic grouping rules** - Parentheses bind tags that describe the same entity
3. **Event classification** - Every event should have both `Event` and `Task-event-role` tags
4. **Data type semantics** - Timeline and descriptor data have different requirements
5. **Relationship patterns** - Agent-action-object and directional relationships need specific structures
6. **Assembly control** - Use curly braces to control how multi-column annotations are assembled
7. **Consistency** - Use the same patterns for similar events throughout your dataset

By following these principles and patterns, you create annotations that are not only syntactically valid but also semantically meaningful and machine-actionable, enabling powerful downstream analysis and cross-study comparisons.

**Additional information**:

- [HED Annotation Quickstart](./HedAnnotationQuickstart.md) - Practical annotation guide
- [BIDS Annotation Quickstart](./BidsAnnotationQuickstart.md) - BIDS integration
- [HED Schemas](./HedSchemas.md) - Understanding the HED vocabulary
- [HED Validation Guide](./HedValidationGuide.md) - Validating your annotations

**Available tools:**

- [HED online tools](https://hedtools.org/hed) - Fairly complete set of tools for a single tsv and json files.
- [HED browser-based validation](https://www.hedtags.org/hed-web) - validate an entire BIDS dataset -- all local, no installation
- [HED extension for NWB](https://github.com/hed-standard/ndx-hed) - incorporates HED into Neurodata Without Borders datasets.
- [Python HEDTools](https://github.com/hed-standard/hed-python) - comprehensive set of tools for HED in Python.
- [MATLAB HEDTools](https://github.com/hed-standard/hed-matlab) - HED interface in MATLAB
