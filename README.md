# PhilBot
//
Created by Joel Z. Hay - UoNPC President and Founder
Custom discord bot for handling the UoNPC discord server. Features games and polls.
//


This bot accesses the Discord and wikipedia API's and features the following:

Automatic polls:
once a week, the bot will post a poll to decide the week's topic. 
After a set amount of time, it will tally the answers and randomly choose a topic from the voted category.
It will then mark that category "used" and exclude it from later searches.

Sentence transformers:
Uses sentence transformers to compare stored topics/topic suggestions and notify the user of any that are semantically similar.

Approve:
Users can suggest topics and admins can approve those topics as options that can later be selected from in polls or debates.

/debate:
Randomly pulls a debate topic from all saved topics, used and unused.

/lookup:
Uses the wikipedia API to provide a one-paragraph summary of anything that can be searched on the site.

/quote:
Provides a random philosophy quote from a pool of 450.

/guess-the-quote:
Sends a quote with no author, and prompts the user to guess who said it.
