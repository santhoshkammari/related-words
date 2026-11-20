"""Hand-written eval: situation (target word NOT used) -> acceptable lemmas. First target = intended word."""
import json, re, sys
S = [
("She stayed perfectly calm and steady even though the meeting turned hostile and everyone else was shouting.", ["composure","poise","equanimity","unflappable"]),
("He kept putting off the report until the night before it was due, even though he had weeks to do it.", ["procrastinate","delay","dawdle","stall"]),
("The inspector went through every line of the contract looking for hidden errors.", ["scrutinize","examine","inspect","scour"]),
("I tried to calm my angry neighbour by offering to pay for the fence he said we damaged.", ["placate","appease","pacify","mollify"]),
("Cutting the budget during a recession only made the economic downturn worse.", ["exacerbate","aggravate","worsen","compound"]),
("The new drainage system should reduce the damage from future floods.", ["mitigate","lessen","alleviate","reduce"]),
("He sat by the window for an hour, thinking carefully about whether to accept the job offer.", ["ponder","contemplate","mull","deliberate"]),
("Gossip and constant leaks slowly weakened the leader's authority from the inside.", ["undermine","erode","sabotage","subvert"]),
("She saves every old newspaper and jar, filling the garage with things she will never use.", ["hoard","stockpile","amass"]),
("The witness made up a story about where he was that night.", ["fabricate","invent","concoct","fake"]),
("The old man mourned the loss of the village he grew up in, repeating how things used to be.", ["lament","mourn","bemoan","grieve"]),
("The coach sharply criticised the player for ignoring the team's instructions.", ["rebuke","reprimand","scold","admonish"]),
("The extra funding strengthened the struggling school's programs and gave teachers new hope.", ["bolster","strengthen","reinforce","buttress"]),
("The high fine was meant to discourage people from dumping rubbish in the park.", ["deter","discourage","dissuade"]),
("The gang forced shopkeepers to pay them money by threatening to wreck their stores.", ["coerce","intimidate","extort","compel"]),
("The suspect managed to avoid capture for three years by moving from town to town.", ["elude","evade","escape","dodge"]),
("From the long report, she managed to pick out the few facts that actually mattered.", ["glean","extract","distill","cull"]),
("The thick fog slowed the rescue teams and made it harder to reach the hikers.", ["hinder","impede","hamper","obstruct"]),
("The accountant checked every figure twice and never left a single detail to chance.", ["meticulous","thorough","painstaking","fastidious"]),
("He never buys anything he does not need and cooks at home to avoid wasting money.", ["frugal","thrifty","economical","prudent"]),
("The instructions could be read in two different ways, so nobody knew what to do.", ["ambiguous","vague","equivocal","unclear"]),
("She did not want to sing in front of everyone and kept finding excuses to stay seated.", ["reluctant","unwilling","hesitant","loath"]),
("Smartphones are everywhere now; you see them in every shop, on every street and in every home.", ["ubiquitous","omnipresent","pervasive","widespread"]),
("He believed everything the stranger told him, even the story about winning a free cruise.", ["gullible","naive","credulous"]),
("Despite losing three times, she refused to give up and kept pushing toward her goal.", ["tenacious","persistent","determined","dogged"]),
("After the heavy meal he felt sluggish and had no energy to do anything but lie on the sofa.", ["lethargic","sluggish","listless","torpid"]),
("She told him honestly and directly what she thought of his plan, without softening anything.", ["candid","frank","blunt","forthright"]),
("He boasted that he was smarter than everyone in the room and looked down on their ideas.", ["arrogant","haughty","conceited","pompous"]),
("Her days were an endless round of dull chores: laundry, dishes, emails, repeat.", ["mundane","humdrum","tedious","monotonous"]),
("The small town bounced back quickly after the factory closed and found new industries.", ["resilient","adaptable","hardy","tough"]),
("The beauty of the cherry blossoms lasts only a few days before the petals fall.", ["ephemeral","fleeting","transient","short-lived"]),
("The wealthy woman gave large amounts of money to hospitals and never asked for credit.", ["benevolent","generous","charitable","philanthropic"]),
("He would not change his mind even after everyone showed him clear evidence he was wrong.", ["obstinate","stubborn","headstrong","intransigent"]),
("The speaker moved the crowd with a graceful and persuasive speech.", ["eloquent","articulate","fluent","persuasive"]),
("When told about the delay, he shrugged and carried on as if it did not matter at all.", ["nonchalant","casual","indifferent","unconcerned"]),
("Their wedding was extravagant, with rare flowers, a huge orchestra and ice sculptures.", ["lavish","opulent","extravagant","sumptuous"]),
("I doubt his promises; I have heard them before and nothing ever happened.", ["skeptical","doubtful","dubious","suspicious"]),
("He could not decide whether to stay with his family or take the dream job abroad; either choice meant losing something.", ["dilemma","quandary","predicament"]),
("Looking at old photos made her long for the summers of her childhood.", ["nostalgia","wistfulness","sentimentality"]),
("During the drought there was too little water and farmers feared the crops would fail.", ["scarcity","shortage","dearth","paucity"]),
("Getting the permit approved was the biggest obstacle standing between us and opening the shop.", ["hurdle","obstacle","barrier","impediment"]),
("After hours of debate, everyone in the committee finally agreed on the same plan.", ["consensus","agreement","accord","unanimity"]),
("Her friendly manner was just an act to hide that she was planning to fire him.", ["pretense","facade","charade","sham"]),
("There is a subtle difference in meaning between the two words that most people miss.", ["nuance","subtlety","distinction"]),
("The jury announced its decision after two days: not guilty.", ["verdict","judgment","ruling"]),
("The error occurred because nobody noticed the missing signature, a mistake from lack of attention.", ["oversight","lapse","omission","negligence"]),
("The two companies had been fighting for customers for decades and neither would back down.", ["rivalry","competition","feud","enmity"]),
("Losing the key player in the first week was a blow that delayed the whole project.", ["setback","reversal","blow","hitch"]),
("Surviving three days lost in the desert without water was a terrible experience she rarely spoke about.", ["ordeal","trial","hardship","nightmare"]),
("The company avoided paying taxes by using a gap in the law that nobody had thought to close.", ["loophole","technicality","gap"]),
("The referee clearly favoured the home team and the visitors never had a fair chance.", ["bias","partiality","prejudice","favoritism"]),
("She has never forgiven her sister and still feels bitter about something said ten years ago.", ["grudge","resentment","rancor","animosity"]),
("He handed in the form without a word, unwilling to say what he really thought.", ["reluctantly","grudgingly","unwillingly"]),
("The door swung open suddenly and everyone in the room stopped talking.", ["abruptly","suddenly","unexpectedly"]),
("I accidentally sent the private message to the whole group without meaning to.", ["inadvertently","accidentally","unintentionally"]),
("He broke the rule on purpose to see how the manager would react.", ["deliberately","intentionally","purposely","knowingly"]),
("The workers carefully and patiently restored the ancient mural, one tiny piece at a time.", ["painstakingly","meticulously","laboriously"]),
("The senator evaded the question by changing the subject three times.", ["equivocate","hedge","dodge","prevaricate"]),
("The cheerful puppy ran in circles, full of lively energy and eager to play.", ["exuberant","playful","frisky","spirited"]),
("The manager praised the team, but the praise was so exaggerated it sounded insincere.", ["effusive","fulsome","gushing","lavish"]),
("The sudden announcement left the whole office stunned and unable to speak.", ["flabbergasted","dumbfounded","astonished","stunned"]),
("After the fight, the two brothers made peace and became friends again.", ["reconcile","make up","rapprochement"]),
("The doctor was careful not to give the patient false hope but did not take away all optimism either.", ["guarded","measured","cautious","tempered"]),
("A strong smell of fresh bread filled the street and made everyone hungry.", ["aroma","scent","fragrance","whiff"]),
]
def leaks(s, w):
    return bool(re.search(r"\b" + re.escape(w[:max(4, len(w) - 2)]), s, re.I))

out, dropped = [], 0
for i, (s, ts) in enumerate(S):
    assert not leaks(s, ts[0]), f"primary target leaks in item {i}: {ts[0]}"
    keep = [w for w in ts if not leaks(s, w)]
    dropped += len(ts) - len(keep)
    out.append((i, s, keep))
with open(sys.argv[1], "w") as f:
    for i, s, ts in out:
        f.write(json.dumps(dict(id=i, situation=s, targets=ts)) + "\n")
print(len(out), "situations written;", dropped, "leaking acceptable targets dropped")
