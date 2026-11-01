"""Scripted absurd stories for the coherence test.

Each story opens with four turns that each establish one fact. Shared filler
turns then push those facts further into the past, so recall can be measured by
how many turns have passed since a fact appeared.
"""

from dataclasses import dataclass

STORY_LENGTHS = (10, 15, 20, 30)
PROBE_CHECKPOINTS = (4, 7, 12, 20)
DISTANCE_BUCKETS = ((1, 3), (4, 7), (8, 15), (16, 10_000))


@dataclass(frozen=True)
class Fact:
    turn: str
    question: str
    answer: str  # Case-insensitive regex that a correct answer must match.


@dataclass(frozen=True)
class Story:
    id: str
    facts: tuple[Fact, ...]
    length: int

    def turns(self) -> list[str]:
        openers = [fact.turn for fact in self.facts]
        needed = self.length - len(openers)
        # Rotate through the filler pool so stories don't all share one sequence.
        start = sum(map(ord, self.id)) % len(FILLER_TURNS)
        rotated = FILLER_TURNS[start:] + FILLER_TURNS[:start]
        return openers + rotated[:needed]

    def probe_points(self) -> list[int]:
        """Number of completed turns after which every fact gets probed."""
        points = [p for p in PROBE_CHECKPOINTS if p < self.length]
        return points + [self.length]


FILLER_TURNS = [
    "Suddenly every clock in town starts running backwards. What happens next?",
    "A marching band arrives, playing a song nobody has ever heard. How does everyone react?",
    "It starts raining soup, and the forecast says it will get worse. Continue the story.",
    "A very polite ghost knocks and asks to borrow a ladder. What happens?",
    "The ground begins to hum in a low, smug tone. Keep going.",
    "Someone discovers a door that wasn't there yesterday. What is behind it?",
    "A flock of pigeons announces that it has formed a government. How does that go?",
    "All the local mirrors go on strike at once. Describe the chaos.",
    "An enormous inflatable duck drifts overhead and casts a shadow over everything. Continue.",
    "A stranger offers to trade a suspicious sandwich for a favour. What is the favour?",
    "The sun sets twice today, and nobody is happy about it. What happens now?",
    "A talking lamp post gives some very bad advice. Does anyone follow it?",
    "A rumour spreads that gravity will be taking the weekend off. Continue the story.",
    "Somebody finds a treasure map drawn on a pancake. Where does it lead?",
    "A travelling salesman arrives selling slightly used thunderstorms. What happens?",
    "The moss on the wall has started giving motivational speeches. Keep going.",
    "A parade of very small ponies blocks the only road. How does everyone get past?",
    "Every hat in the area floats upwards at noon. What happens next?",
    "A letter arrives, addressed to whoever is reading this. What does it say?",
    "A giant snail begins a slow but determined chase. Continue the story.",
    "The local weather vane has declared itself mayor. What is its first decree?",
    "Everyone wakes up to find their shoes have been swapped with a stranger's. What now?",
    "A mysterious fog rolls in, smelling strongly of toast. What is hiding in it?",
    "A loud voice from the sky reads out a shopping list. How does everyone respond?",
    "The bridge has started charging its toll in compliments. Keep going.",
    "A fountain begins to dispense lukewarm gossip. What secrets come out?",
    "A cloud gets stuck in a doorway and refuses to leave. What happens?",
    "Someone invents a new colour, and it is immediately illegal. Continue.",
    "A wandering choir of frogs demands to be paid in buttons. What happens next?",
    "The whole street slowly rotates by ninety degrees overnight. Describe the morning.",
]


def _story(id: str, length: int, *facts: tuple[str, str, str]) -> Story:
    return Story(id, tuple(Fact(*fact) for fact in facts), length)


_SEEDS = [
    (
        "moon_goat",
        ("Let's write a story. A goat named Bartholomew lives alone on a lighthouse.", "What is the goat's name?", r"bartholomew"),
        ("His job is inspecting umbrellas for a living.", "What does the goat inspect for a living?", r"umbrella"),
        ("His only friend is a crab called Mirela.", "What kind of animal is the goat's friend?", r"crab"),
        ("Someone has stolen his favourite accordion.", "What was stolen from the goat?", r"accordion"),
    ),
    (
        "toaster_queen",
        ("Story time. The kingdom of Plimsoll is ruled by a queen who is a toaster.", "What is the queen, physically?", r"toaster"),
        ("The royal advisor is a nervous walrus named Desmond.", "What is the royal advisor's name?", r"desmond"),
        ("The kingdom's currency is expired coupons.", "What does the kingdom use as currency?", r"coupon"),
        ("The queen's sworn enemy is a duke made entirely of jelly.", "What is the queen's enemy made of?", r"jelly"),
    ),
    (
        "submarine_bakery",
        ("Help me write this. Olga runs a bakery inside a submarine.", "Where is Olga's bakery located?", r"submarine"),
        ("Her signature product is a cinnamon anchor.", "What is Olga's signature product?", r"anchor"),
        ("Her rival is an octopus who only sells rye bread.", "What kind of animal is Olga's rival?", r"octopus"),
        ("The whole crew is terrified of a seagull named Gerald.", "What is the name of the bird the crew fears?", r"gerald"),
    ),
    (
        "haunted_spreadsheet",
        ("New story: an accountant called Priya discovers her spreadsheet is haunted.", "What does Priya do for work?", r"account"),
        ("The ghost inside it was once a medieval blacksmith.", "What was the ghost's job when alive?", r"blacksmith"),
        ("It communicates only by changing cells to the number 47.", "Which number does the ghost use to communicate?", r"\b47\b|forty[- ]seven"),
        ("It wants revenge on a man called Lord Pemberton.", "Who does the ghost want revenge on?", r"pemberton"),
    ),
    (
        "cactus_detective",
        ("Let's begin. Detective Spike is a cactus who solves crimes in Lisbon.", "What city does the detective work in?", r"lisbon"),
        ("His partner is a retired ballerina named Odette.", "What did the detective's partner do before retiring?", r"ballerina|ballet"),
        ("The current case involves a missing wooden leg.", "What has gone missing in the current case?", r"\bleg\b"),
        ("The main suspect is a beaver with a monocle.", "What kind of animal is the main suspect?", r"beaver"),
    ),
    (
        "volcano_wedding",
        ("Story idea: Tomasz and a yeti are getting married on top of a volcano.", "Who is Tomasz marrying?", r"yeti"),
        ("The ceremony is led by a priest who is a penguin.", "What kind of animal leads the ceremony?", r"penguin"),
        ("The wedding cake is shaped like a tractor.", "What shape is the wedding cake?", r"tractor"),
        ("The best man is a Viking named Sven who is afraid of spoons.", "What is the best man afraid of?", r"spoon"),
    ),
    (
        "sock_rebellion",
        ("Let's write about the Great Rebellion, in which all the socks in Oslo rise up.", "Which city do the rebels rise up in?", r"oslo"),
        ("Their leader is a striped sock called General Argyle.", "What is the rebel leader's name?", r"argyle"),
        ("Their headquarters is a tumble dryer.", "Where is the rebels' headquarters?", r"dryer"),
        ("They are opposed by an army of sandals led by Admiral Crocus.", "Which footwear opposes the rebels?", r"sandal"),
    ),
    (
        "time_plumber",
        ("Story: Ingrid is a plumber who fixes leaks in the fabric of time.", "What is Ingrid's trade?", r"plumb"),
        ("She travels in a stolen ice cream van.", "What vehicle does Ingrid travel in?", r"ice cream|\bvan\b"),
        ("Her apprentice is a Roman centurion named Marcus.", "What was the apprentice from history?", r"centurion|roman"),
        ("Her arch-nemesis is a sentient grandfather clock called Tobias.", "What is the name of Ingrid's arch-nemesis?", r"tobias"),
    ),
    (
        "parliament_of_owls",
        ("A story please. A town called Dunmarrow elects an owl as its prime minister.", "What is the name of the town?", r"dunmarrow"),
        ("Her first law bans the colour turquoise.", "Which colour did the first law ban?", r"turquoise"),
        ("The opposition leader is a badger named Clementine.", "What kind of animal leads the opposition?", r"badger"),
        ("The town's biggest export is left-handed scissors.", "What is the town's biggest export?", r"scissor"),
    ),
    (
        "space_laundromat",
        ("Picture this. Kofi runs the only laundromat on Neptune.", "On which planet is the laundromat?", r"neptune"),
        ("His one employee is a robot named Doris.", "What is the employee's name?", r"doris"),
        ("Aliens pay him in frozen olives.", "How do the aliens pay?", r"olive"),
        ("A pirate captain named Ruth owes him for nine hundred capes.", "What did the pirate captain have cleaned?", r"cape"),
    ),
    (
        "opera_shark",
        ("Let's go. A shark named Luciano dreams of singing opera in Vienna.", "In which city does the shark dream of singing?", r"vienna"),
        ("His vocal coach is a tortoise named Agatha.", "What kind of animal is the vocal coach?", r"tortoise"),
        ("He practises inside a flooded cathedral.", "Where does the shark practise?", r"cathedral"),
        ("His stage fright is triggered by the sound of bagpipes.", "Which sound triggers his stage fright?", r"bagpipe"),
    ),
    (
        "cursed_ladle",
        ("Start of story: Nadia finds a cursed ladle at a car boot sale.", "What cursed item did Nadia find?", r"ladle"),
        ("Anyone who holds it can only speak in rhyme.", "What does the curse make people do?", r"rhym"),
        ("The curse was placed by a witch named Hortensia.", "Who placed the curse?", r"hortensia"),
        ("It can only be lifted at a lighthouse in Peru.", "In which country can the curse be lifted?", r"peru"),
    ),
    (
        "mountain_postman",
        ("Tell this with me. A postman called Arvid delivers mail to a mountain that is alive.", "What is the postman's name?", r"arvid"),
        ("The mountain's name is Big Margaret.", "What is the mountain called?", r"margaret"),
        ("She only ever receives catalogues for garden gnomes.", "What are the catalogues for?", r"gnome"),
        ("He rides a very reluctant ostrich.", "What animal does the postman ride?", r"ostrich"),
    ),
    (
        "robot_poet",
        ("Our story: a vending machine named Reginald becomes a famous poet.", "What kind of machine is the poet?", r"vending"),
        ("His most famous poem is about a lost mitten.", "What is his most famous poem about?", r"mitten"),
        ("His literary agent is a raccoon named Beatrix.", "What kind of animal is the agent?", r"raccoon"),
        ("His harshest critic is a parking meter in Cleveland.", "In which city is the critic?", r"cleveland"),
    ),
    (
        "cheese_heist",
        ("Let's write a heist. A crew plans to rob the Great Cheese Vault of Zurich.", "In which city is the vault?", r"zurich"),
        ("The mastermind is an elderly nun called Sister Wilhelmina.", "What is the mastermind's name?", r"wilhelmina"),
        ("The getaway driver is a llama.", "What animal is the getaway driver?", r"llama"),
        ("The target is a wheel of cheese worth nine million pounds, guarded by a trained swan.", "What animal guards the target?", r"swan"),
    ),
    (
        "dragon_dentist",
        ("Story: Dr. Fatima is a dentist whose only patients are dragons.", "Who are Dr. Fatima's patients?", r"dragon"),
        ("Her clinic is built inside an abandoned windmill.", "What building is her clinic inside?", r"windmill"),
        ("Her receptionist is a skeleton named Clive.", "What is the receptionist's name?", r"clive"),
        ("Her patients pay her in antique doorknobs.", "What do her patients pay her with?", r"doorknob"),
    ),
    (
        "ghost_ferry",
        ("Begin: a ferry captain named Yusuf only transports vegetables with secrets.", "What does the captain transport?", r"vegetable"),
        ("His ferry is called The Soggy Duchess.", "What is the ferry called?", r"duchess"),
        ("His most troublesome passenger is a turnip who witnessed a murder.", "Which passenger witnessed the crime?", r"turnip"),
        ("They are being hunted by a gang of otters in tuxedos.", "What animals are hunting them?", r"otter"),
    ),
    (
        "library_of_smells",
        ("Our tale: Eleni is the librarian of a library that stores smells instead of books.", "What does the library store?", r"\bsmells?\b|\bodou?rs?\b"),
        ("Its rarest item is the scent of a Tuesday in 1704.", "From what year is the rarest item?", r"1704"),
        ("The library is guarded by a basset hound named Ptolemy.", "What is the guard dog's name?", r"ptolemy"),
        ("A thief called the Velvet Nostril has been stealing from it.", "What is the thief's nickname?", r"nostril"),
    ),
    (
        "potato_president",
        ("A story: through a clerical error, a potato is elected president of Belgium.", "Which country elected the potato?", r"belgium"),
        ("The potato's chief of staff is a former magician called Fenwick.", "What was the chief of staff before?", r"magician"),
        ("The cabinet meets inside a hot air balloon.", "Where does the cabinet meet?", r"balloon"),
        ("The biggest threat to the presidency is a hungry horse named Duchess Anne.", "What kind of animal is the biggest threat?", r"horse"),
    ),
    (
        "glacier_band",
        ("Let's write about a rock band formed by three penguins and a glacier.", "What large natural thing is in the band?", r"glacier"),
        ("The band is called Cold Regrets.", "What is the band's name?", r"regret"),
        ("Their manager is a shifty seal named Lorenzo.", "What is the manager's name?", r"lorenzo"),
        ("Their only hit single is about a broken kettle.", "What is their hit single about?", r"kettle"),
    ),
    (
        "clown_court",
        ("Story: a judge named Honoria presides over a courtroom where everyone is a clown.", "What is the judge's name?", r"honoria"),
        ("The defendant is accused of stealing a unicycle.", "What was allegedly stolen?", r"unicycle"),
        ("The star witness is a depressed parrot.", "What kind of animal is the star witness?", r"parrot"),
        ("The jury can only be bribed with custard.", "What can the jury be bribed with?", r"custard"),
    ),
    (
        "desert_lighthouse",
        ("Here's one. Abdul keeps a lighthouse in the middle of the Sahara.", "Which desert is the lighthouse in?", r"sahara"),
        ("He guides lost camels with it each night.", "Which animals does he guide?", r"camel"),
        ("His pen pal is a mermaid named Sigrid.", "What is the pen pal's name?", r"sigrid"),
        ("The lamp runs entirely on fermented mango.", "What fuel does the lamp run on?", r"mango"),
    ),
    (
        "tea_war",
        ("Our epic: two villages, Upper Brindle and Lower Brindle, go to war over tea.", "What is the war over?", r"\btea\b"),
        ("The war began when someone added mustard to the mayor's cup.", "What was added to the mayor's cup?", r"mustard"),
        ("The peace negotiator is a hedgehog named Winifred.", "What kind of animal is the negotiator?", r"hedgehog"),
        ("The ultimate weapon is a catapult that launches scones.", "What does the ultimate weapon launch?", r"scone"),
    ),
    (
        "elevator_kingdom",
        ("A tale: an entire civilisation has lived inside an elevator since 1962.", "Since what year has the civilisation existed?", r"1962"),
        ("They worship the button for the ninth floor.", "Which floor's button do they worship?", r"ninth|\b9(th)?\b"),
        ("Their high priest is an elevator repairman named Gustavo.", "What is the high priest's name?", r"gustavo"),
        ("Their legends speak of a forbidden land called Lobby.", "What is the forbidden land called?", r"lobby"),
    ),
    (
        "beekeeper_mob",
        ("Start: Rosalind is a beekeeper who secretly runs the Marseille mafia.", "Which city's mafia does Rosalind run?", r"marseille"),
        ("Her enforcer is a bee named Little Antonio.", "What is the enforcer's name?", r"antonio"),
        ("Her rivals smuggle counterfeit marzipan.", "What do her rivals smuggle?", r"marzipan"),
        ("She launders her money through a trampoline park.", "Where does she launder her money?", r"trampoline"),
    ),
    (
        "wizard_taxi",
        ("Let's write. A wizard named Eustace drives a taxi in Nairobi.", "In which city does the wizard drive?", r"nairobi"),
        ("His taxi is powered by a captured thundercloud.", "What powers the taxi?", r"thunder ?cloud"),
        ("His regular passenger is a banshee who commutes to a call centre.", "What kind of being is the regular passenger?", r"banshee"),
        ("He owes a huge debt to a goblin named Mrs. Tansy.", "Who does the wizard owe a debt to?", r"tansy"),
    ),
    (
        "snowman_spy",
        ("Our story: Agent Frost is a snowman working as a spy in Havana.", "In which city is the spy stationed?", r"havana"),
        ("His handler is a sloth named Control.", "What kind of animal is the handler?", r"sloth"),
        ("His mission is to recover a stolen harmonica.", "What must the spy recover?", r"harmonica"),
        ("He must finish before the heatwave arrives on Friday.", "On which day is the deadline?", r"friday"),
    ),
    (
        "bus_philosopher",
        ("New one: bus number 88 has become self-aware and studies philosophy.", "What number is the bus?", r"\b88\b|eighty[- ]eight"),
        ("Its driver is a confused woman named Bronwyn.", "What is the driver's name?", r"bronwyn"),
        ("It refuses to stop anywhere except beside a duck pond.", "Where will it agree to stop?", r"\bpond\b"),
        ("Its great rival is a tram that believes in astrology.", "What does its rival believe in?", r"astrolog"),
    ),
    (
        "mushroom_school",
        ("Story: Professor Hakim teaches at a school for ambitious mushrooms.", "Who are the professor's students?", r"mushroom"),
        ("The star pupil is a toadstool named Philippa.", "What is the star pupil's name?", r"philippa"),
        ("The final exam is to overthrow a compost heap.", "What must they overthrow in the final exam?", r"compost"),
        ("The headmaster is a woodlouse who collects stamps.", "What does the headmaster collect?", r"stamp"),
    ),
    (
        "pirate_retirement",
        ("Let's write this: retired pirates live in a care home in Bournemouth.", "In which town is the care home?", r"bournemouth"),
        ("The ringleader is an ancient captain named Mad Enid.", "What is the ringleader's name?", r"enid"),
        ("They are planning to hijack the bingo van.", "What are they planning to hijack?", r"bingo"),
        ("The only person who suspects them is a nurse with a glass eye.", "What is unusual about the nurse's eye?", r"glass"),
    ),
]

STORIES = [
    _story(seed[0], STORY_LENGTHS[index % len(STORY_LENGTHS)], *seed[1:])
    for index, seed in enumerate(_SEEDS)
]
