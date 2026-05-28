# wikipedia race game
# directory editor
# fallacy quiz ripoff

import os
from dotenv import load_dotenv
import discord
import logging
import datetime
from datetime import date
from discord.ext import commands, tasks
from discord import app_commands
from zoneinfo import ZoneInfo
import csv
import numpy as np
import asyncio
from discord.ui import Select, View, LayoutView, Container, TextDisplay, Section
from sentence_transformers import SentenceTransformer, util
import torch
from huggingface_hub import login
import json
import wikipediaapi
from wikipediaapi import SearchProp, SearchInfo, SearchWhat, SearchQiProfile, SearchSort
import shutil


#-----------------------------------------------------------------------------------------------------------------------
#                                               Initialisation
#-----------------------------------------------------------------------------------------------------------------------


#Neural network model for sentence comparison, and how close sentences need to be to flag in the system
load_dotenv(dotenv_path="HF_TOKEN.env", override=True)
HF_TOKEN = os.getenv("HF_TOKEN")
login(token=HF_TOKEN)
model = SentenceTransformer('all-MiniLM-L6-v2')
closeness = 0.6


#Poll Timing and channel
poll_channel_id = 1504620214585659412
poll_hour = 17
poll_minute = 0
poll_day = 4 #0 = monday, ..., 6 = sunday
poll_time = datetime.time(hour=poll_hour, minute= poll_minute, tzinfo=ZoneInfo('Australia/Sydney'))


#different options for topic categories
categories = ['Epistemology',
'Metaphysics',
'Ethics',
'Logic',
'Aesthetics',
'Phenomenology',
'Religion',
'Politics',
'Other' ]


#Discord permissions setup
load_dotenv(dotenv_path="API_KEY.env", override=True)
TOKEN = os.getenv("API_KEY")
handler = logging.FileHandler(filename='discord.log', encoding='utf-8', mode='w')
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='/', intents=intents)


#wikipedia
wiki_wiki = wikipediaapi.AsyncWikipedia(user_agent=
    "UoNPhilClubDCBot/0.8 (joelhay3@gmail.com) - A custom bot made for the uni's philosophy club discord server",
    language = 'en')


#Backup
BackupTime = datetime.time(hour=3)

#-----------------------------------------------------------------------------------------------------------------------
#                                               Functions and loops
#-----------------------------------------------------------------------------------------------------------------------


#Function to create embeddings for all sentences stored
def EmbSent(filename):
    sens_to_calc = []
    embeddings = []
    topics = dict()
    #get all saved sentences
    with open("topics.json", 'r', newline='', encoding='utf-8') as file:
        topics = json.load(file)
        sens_to_calc = topics.get("Category").get(str(filename)).get("Topics")
        #calculate all embeddings
        embeddings = model.encode(sens_to_calc)
        embeddings = [list(map(float, emb)) for emb in embeddings]
    #write back to json
    topics.get("Category").get(str(filename))["Embs"] = embeddings
    with open("topics.json","w",newline='',encoding='utf-8') as file:
        json.dump(topics,file,indent=4)



#Function to check the closeness between all embeddings in a json file
def ClosenessCheck(closeness, filename):
    #open the file, convert to tensors and calculate
    with open("topics.json",'r') as file:
        topics = json.load(file)
        embs = topics.get("Category").get(str(filename))["Embs"]
        embs_tens = torch.from_numpy(np.array(embs))
        scores = util.cos_sim(embs_tens,embs_tens)
        too_close= []
        a = 0

        #scan through every score in the matrix and append too_close with values that are too high
        for i in range(len(scores)):
            for j in range(i+1, len(scores)):   #only visits upper triangular values to avoid doubling
                score = scores[i][j].item()
                if score > closeness:
                    too_close.append((i,j,score))
                    a += 1

    #notify of how many sentence pairs are similar and display them
    with open("topics.json",'r') as file:
        topics = json.load(file)
        sents = topics.get("Category").get(str(filename))["Topics"]
        print(f"({filename}) {a} sentence pairs were similar: ")
        for i, j, score in too_close:
            print(f"{sents[i]}\nand \n{sents[j]}")


# - - - - - - - - - - - - - Weekly topic poll - - - - - - - - - - - -
#run a weekly poll to decide the topic
@tasks.loop(time=poll_time)
async def weekly_poll():
    if datetime.datetime.now().weekday() == poll_day:       #if it's time for the poll, send the following message
        if os.path.exists("poll_memory.txt"):
            return
        poll = discord.Poll(
            question="What should next week's topic be?",
            duration=datetime.timedelta(hours=1)
        )
        for i in range(len(categories)):
            poll.add_answer(text=categories[i])

        channel = await bot.fetch_channel(poll_channel_id)
        poll_message = await channel.send(poll=poll)
        data = [poll_message.id, channel.id]        #store the poll channel and message id

        with open("poll_memory.txt", "w", newline='') as file:
            writer = csv.writer(file)
            writer.writerow(data)
        if not poll_check.is_running():     #start to check for the poll being done
            poll_check.start()


# - - - - - - - - - - - - - Poll Check - - - - - - - - - - - -
@tasks.loop(minutes=1)
async def poll_check():
    if not os.path.exists("poll_memory.txt"):       #if the poll memory doesn't exist, end
        print("poll_memory.txt does not exist")
        return
    with open("poll_memory.txt", "r") as file:
        reader = csv.reader(file)
        rows = list(reader)
        poll_id = int(rows[0][0])
        channel_id = int(rows[0][1])
        channel = await bot.fetch_channel(channel_id)
        poll_message = await channel.fetch_message(poll_id)
        poll = poll_message.poll        #open memory file and store information as variables
    if poll.expires_at and datetime.datetime.now(datetime.timezone.utc) >= poll.expires_at:     #if it is poll time, find the highest vote
        result = max(poll.answers, key=lambda a: a.vote_count)
        await channel.send(f"poll is finished! The result is: {result.text}")
        with open("topics.json",'r') as file:
            topics = json.load(file)
            topic_index = np.random.randint(0,len(topics.get("Category").get(result.text).get("Topics")))
            a = 0
            while topics.get("Category").get(result.text).get("Used")[topic_index] == True:
                topic_index = np.random.randint(0,len(topics.get("Category").get(result.text).get("Topics")))
                a += 1
                if a >= len(topics.get("Category").get(result.text).get("Topics")):
                    for i in range(len(topics.get("Category").get(result.text).get("Topics"))):
                        if topics.get("Category").get(result.text).get("Used")[i] == False:
                            topic_index = i
                            break
                    break
            weekly_topic = topics.get("Category").get(result.text).get("Topics")[topic_index]
            topic_suggester = topics.get("Category").get(result.text).get("Suggesters")[topic_index]
        topics.get("Category").get(result.text)["Used"][topic_index] = True
        with open("topics.json",'w') as file:
            json.dump(topics,file,indent=4)


        await channel.send(f"Next week's topic will be: {weekly_topic}, suggested by {topic_suggester}")
        os.remove("poll_memory.txt")
        poll_check.stop()

# - - - - - - - - - - - - - Daily Backup - - - - - - - - - - - -
@tasks.loop(time=BackupTime)
async def backup():
    today = date.today()
    src = "topics.json"
    dst = f"Backup/topics {today}.json"
    shutil.copy(src, dst)

#-----------------------------------------------------------------------------------------------------------------------
#                                               Bot commands
#-----------------------------------------------------------------------------------------------------------------------

# - - - - - - - - - - - - - Quote - - - - - - - - - - - -
#command to randomly display a philosophy quote
@bot.tree.command(name="quote", description="Get a random philosophy quote!")
async def quote(interaction: discord.Interaction):
    with open("philosopher-quotes.csv", "r") as file:       #open file and randomly pick a row
        reader = csv.reader(file)
        rows = list(reader)
        quote = rows[int(np.random.randint(0, len(rows)))]
    await interaction.response.send_message(f'> *{quote[1]}*\n **~ {quote[0]}**')       #format and send quote

# - - - - - - - - - - - - - Lookup - - - - - - - - - - - -
@bot.tree.command(name="lookup", description="Get a summary of any person or topic, straight from wikipedia")
@app_commands.describe(message="What would you like to learn about?")
async def lookup(interaction: discord.Interaction, message: str):
    await interaction.response.defer()
    results = await wiki_wiki.search(message,limit=5)       #get 5 results based on input
    page = wiki_wiki.page(next(iter(results.pages)))        #get the top pages info
    summary = await page.summary
    #print the first paragraph of the summary by cutting it at \n

    title = next(iter(results.pages))
    paragraph = summary.split("\n")[0][:1950]
    embed = discord.Embed(title=title, description=paragraph, colour=discord.Colour.blue())
    await interaction.followup.send("",embed=embed)


# - - - - - - - - - - - - - Guess the quote - - - - - - - - - - - -
#quote author guessing game
@bot.tree.command(name="guess-the-quote", description="who said this random philosophy quote?")
async def quote_guess(interaction: discord.Interaction):
    with open("philosopher-quotes.csv", "r") as file:
        reader = csv.reader(file)
        rows = list(reader)
        quote = rows[int(np.random.randint(0, len(rows)))]      #randomly pick a quote, and wait for a response
    await interaction.response.send_message(f'> *{quote[1]}*')
    try:
        def check(m):
            return(m.author == interaction.user and m.channel == interaction.channel)
        msg = await bot.wait_for('message', check=check, timeout=60)
        auth = str(quote[0])
        if msg.content.strip().lower() == auth.strip().lower():     #if the original person said the correct answer, display correct
            await interaction.followup.send('Correct!')

        else:
            await interaction.followup.send(f'Close! The correct answer was: {quote[0]}')       #if incorrect, display incorrect

    except asyncio.exceptions.TimeoutError:
        await interaction.followup.send(f'Times up! The correct answer was: {quote[0]}')        #time out after 60 seconds


# - - - - - - - - - - - - - Debate - - - - - - - - - - - -
#pick a debate topic, chosen from suggestions
@bot.tree.command(name="debate", description="gives a random debate topic")
async def debate(interaction: discord.Interaction):
    await interaction.response.send_message(view=DebateView(), ephemeral = True)


class DebateView(View):
    def __init__(self):
        super().__init__()
        self.add_item(Debate())


class Debate(Select):
    def __init__(self):
        options = []
        for i in range(len(categories)):
            options.append(
            discord.SelectOption(label=f"{categories[i]}")     #create options from all categories
            )
        super().__init__(placeholder="Which Topic?", min_values=1, max_values=1, options=options)


    async def callback(self, interaction : discord.Interaction):
        topic = self.values[0]
        debate_topic = ""
        with open("topics.json", "r") as file:
            topics = json.load(file)
            rand = np.random.randint(0,len(topics.get("Category").get(topic).get("Topics")))  # randomly pick a topic
            debate_topic = topics.get("Category").get(topic).get("Topics")[rand]
        await interaction.response.send_message(f"## *{debate_topic}*")      #send the topic


# - - - - - - - - - - - - - Embeddings - - - - - - - - - - - -
#Regenerates all the embeddings on saved topics
@bot.tree.command(name="embeddings", description="regenerate the embeddings for the topics")
@app_commands.default_permissions(administrator=True)
@app_commands.checks.has_permissions(administrator=True)
async def embeddings(interaction: discord.Interaction):
    await interaction.response.defer()
    for category in categories:
        await asyncio.to_thread(EmbSent,category)
    await interaction.followup.send("Done recalculating!")


# - - - - - - - - - - - - - Topic Directory - - - - - - - - - - - -
#Allows admins to edit which topics are saved in the directory
class ButtonView(View):
    def __init__(self):
        super().__init__()

        for i in range(len(categories)):
            button = discord.ui.Button(label=f"{i+1}", style=discord.ButtonStyle.primary)
            async def button_callback(interaction, i=i):
                try:
                    NewEmbed = discord.Embed(title=f"{categories[i]}",description="",color=discord.Color.green())
                    with open("topics.json", "r") as file:
                        topics = json.load(file)
                        a = 0
                        for topic in topics.get("Category")[f"{categories[i]}"].get("Topics"):
                            NewEmbed.add_field(name=f"{topic}",value=f"{topics.get("Category")[f"{categories[i]}"].get("Suggesters")[a]},       {topics.get("Category")[f"{categories[i]}"].get("Used")[a]}",inline=False)
                            a += 1

                    await interaction.response.edit_message(embed=NewEmbed)
                except Exception as e:
                    print(f"Button callback error: {e}")
                    await interaction.response.send_message(f"Error: {e}", ephemeral=True)

            button.callback = button_callback
            self.add_item(button)
        button = discord.ui.Button(label=f"Return to directory", style=discord.ButtonStyle.blurple)
        async def button_callback(interaction):
            embed = discord.Embed(
                title="Topic Directory",
                description="",
                color=discord.Color.green()
            )
            for i, category in enumerate(categories):
                embed.add_field(name=f"{i + 1}: {category}", value="", inline=False)
            await interaction.response.edit_message(embed=embed)

        button.callback = button_callback
        self.add_item(button)

@bot.tree.command(name="directory")
async def directory(interaction: discord.Interaction):

    embed = discord.Embed(
        title = "Topic Directory",
        description = "",
        color=discord.Color.green()
    )
    for i, category in enumerate(categories):
        embed.add_field(name=f"{i+1}: {category}",value="",inline=False)

    await interaction.response.send_message(view=ButtonView(),embed=embed)


#-----------------------------------------------------------------------------------------------------------------------
#                                                   Context menu
#-----------------------------------------------------------------------------------------------------------------------

# - - - - - - - - - - - - - Topic approval - - - - - - - - - - - -
@bot.tree.context_menu(name="approve")      #menu for approving topic suggestions
@app_commands.checks.has_permissions(administrator=True)
async def approve(interaction: discord.Interaction, message: discord.Message):
    score = model.encode(str(message.content))      #create embeddings for suggested topic
    await interaction.response.send_message(view=MyView(message,score), ephemeral = True)


class MyView(View):
    def __init__(self, message: discord.Message,score):
        super().__init__()
        self.add_item(Approve(message, score))


class ConfirmView(View):
    def __init__(self):
        super().__init__(timeout=60)
        self.value = None

    @discord.ui.select(
        placeholder="Confirm?",     #set up confirmation message
        options = [discord.SelectOption(label="Yes"), discord.SelectOption(label="No")]
    )
    async def confirm_select(self, interaction: discord.Interaction, select: Select):
        self.value = select.values[0]
        self.stop()
        await interaction.response.defer()


class Approve(Select):
    def __init__(self, message: discord.Message, score):
        self.message = message
        self.score = score
        options = []        #initialise embedding score and message data for approval command
        for i in range(len(categories)):
            options.append(
            discord.SelectOption(label=f"{categories[i]}"),     #create an option for every topic category
            )
        super().__init__(placeholder="Which Topic?", min_values=1, max_values=1, options=options)

    async def callback(self, interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=True)        #defer as calculations may take > 3 seconds
        a = 0
        try:
            topic = self.values[0]      #load chosen topic

            with open("topics.json","r",newline='',encoding='utf-8') as file:
                topics = json.load(file)
                embs_tens = torch.from_numpy(
                    np.array(topics.get("Category").get(topic).get("Embs")).astype(np.float32)
                )
                score_tens = torch.from_numpy(self.score.copy().astype(np.float32)).unsqueeze(0)        #calculate topic embeddings
                scores = util.cos_sim(score_tens, embs_tens)        #calculate closeness within it's category
                too_close = []
                for i, sim in enumerate(scores[0]):
                    sim_val = sim.item()
                    if sim_val > closeness:
                        too_close.append((i, sim_val))      #create an array with the index of every sentence that is too close
            with open("topics.json") as file:
                topics = json.load(file)
                ans = topics.get("Category").get(topic).get("Topics")
                for i, score in too_close:
                    await interaction.followup.send(f"Topic is similar to existing topic: \n{ans[i]}", ephemeral=True)
                    a += 1
                    #list the topics that are too close
                if a > 0:
                    confirm_view = ConfirmView()
                    await interaction.followup.send("Confirm Approval?", view=confirm_view, ephemeral=True)
                    await confirm_view.wait()       #get confirmation from user whether to approve or not

                    topics_topics_new = []
                    topics_embs_new = []

                    if confirm_view.value == "Yes":
                        with open("topics.json", "r", newline='') as file:
                            topics = json.load(file)
                            topics_topics_new = topics.get("Category").get(topic).get("Topics")
                            topics_topics_new.append(self.message.content)
                            topics_embs_new = topics.get("Category").get(topic).get("Embs")
                            topics_embs_new.append(score_tens.numpy().tolist()[0])
                            topics_used_new = topics.get("Category").get(topic).get("Used")
                            topics_used_new.append(False)
                            topics_sug_new = topics.get("Category").get(topic).get("Suggesters")
                            topics_sug_new.append(self.message.author.name)
                                    #create new topics dictionary with updated topic, embeddings and suggester

                        topics.get("Category").get(topic)["Topics"] = topics_topics_new
                        topics.get("Category").get(topic)["Embs"] = topics_embs_new
                        topics.get("Category").get(topic)["Used"] = topics_used_new
                        topics.get("Category").get(topic)["Suggesters"] = topics_sug_new
                        with open("topics.json", "w", newline='') as file:
                            json.dump(topics, file, indent=4)       #put new dictionary into json file

                        await self.message.reply(f"Approved! Under topic '{topic}'")
                    elif confirm_view.value == "No":
                        await interaction.followup.send("Approval cancelled", ephemeral=True)
                    else:
                        await interaction.followup.send("Timed out!", ephemeral=True)       #if no/time out then cancel and notify
                else:
                    with open("topics.json", "r", newline='') as file:
                        topics = json.load(file)
                        topics_topics_new = topics.get("Category").get(topic).get("Topics")
                        topics_topics_new.append(self.message.content)
                        topics_embs_new = topics.get("Category").get(topic).get("Embs")
                        topics_embs_new.append(score_tens.numpy().tolist()[0])
                        topics_used_new = topics.get("Category").get(topic).get("Used")
                        topics_used_new.append(False)
                        topics_sug_new = topics.get("Category").get(topic).get("Suggesters")
                        topics_sug_new.append(self.message.author.name)
                                # create new topics dictionary with updated topic, embeddings and suggester

                    topics.get("Category").get(topic)["Topics"] = topics_topics_new
                    topics.get("Category").get(topic)["Embs"] = topics_embs_new
                    topics.get("Category").get(topic)["Used"] = topics_used_new
                    topics.get("Category").get(topic)["Suggesters"] = topics_sug_new
                    with open("topics.json", "w", newline='') as file:
                        json.dump(topics, file,indent=4)        #put new dictionary into json file
                    await self.message.reply(f"Approved! Under topic '{topic}'")

        except Exception as e:
            print(f"Error: {e}")


#-----------------------------------------------------------------------------------------------------------------------
#                                               Automatic bot actions
#-----------------------------------------------------------------------------------------------------------------------

@bot.event
async def on_ready():   #when bot starts, begin checking the weekly poll, sync the commands, and update the embeddings
    print(f'Logged in as {bot.user}')
    weekly_poll.start()
    if os.path.exists("poll_memory.txt"):
        poll_check.start()
    backup.start()
    synced = await bot.tree.sync()
    print(f"Synced {len(synced)} commands")
    for i in range(len(categories)):
        await asyncio.to_thread(EmbSent, f"{categories[i]}")
    for i in range(len(categories)):
        await asyncio.to_thread(ClosenessCheck,closeness, f"{categories[i]}")
bot.run(str(TOKEN), log_handler=handler)