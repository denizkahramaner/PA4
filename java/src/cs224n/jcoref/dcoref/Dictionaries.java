package edu.stanford.nlp.jcoref.dcoref;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.Set;

import edu.stanford.nlp.io.IOUtils;
import edu.stanford.nlp.io.RuntimeIOException;
import edu.stanford.nlp.jcoref.docclustering.TfIdf;
import edu.stanford.nlp.pipeline.DefaultPaths;
import edu.stanford.nlp.stats.ClassicCounter;
import edu.stanford.nlp.stats.Counter;
import edu.stanford.nlp.util.Pair;

public class Dictionaries {

  public static final int THESAURUS_THRESHOLD = 10;
  public static final double THESAURUS_SCORE_THRESHOLD = 0.1;

  public enum MentionType { PRONOMINAL, NOMINAL, PROPER }

  public enum Gender { MALE, FEMALE, NEUTRAL, UNKNOWN }

  public enum Number { SINGULAR, PLURAL, UNKNOWN }
  public enum Animacy { ANIMATE, INANIMATE, UNKNOWN }
  public enum Person { I, YOU, HE, SHE, WE, THEY, IT, UNKNOWN}

  public final Set<String> lightVerb = new HashSet<String>(Arrays.asList("have", "make", "get", "take", "receive"));
  public final Set<String> reportVerb = new HashSet<String>(Arrays.asList(
      "accuse", "acknowledge", "add", "admit", "advise", "agree", "alert",
      "allege", "announce", "answer", "apologize", "argue",
      "ask", "assert", "assure", "beg", "blame", "boast",
      "caution", "charge", "cite", "claim", "clarify", "command", "comment",
      "compare", "complain", "concede", "conclude", "confirm", "confront", "congratulate",
      "contend", "contradict", "convey", "counter", "criticize",
      "debate", "decide", "declare", "defend", "demand", "demonstrate", "deny",
      "describe", "determine", "disagree", "disclose", "discount", "discover", "discuss",
      "dismiss", "dispute", "disregard", "doubt", "emphasize", "encourage", "endorse",
      "equate", "estimate", "expect", "explain", "express", "extoll", "fear", "feel",
      "find", "forbid", "forecast", "foretell", "forget", "gather", "guarantee", "guess",
      "hear", "hint", "hope", "illustrate", "imagine", "imply", "indicate", "inform",
      "insert", "insist", "instruct", "interpret", "interview", "invite", "issue",
      "justify", "learn", "maintain", "mean", "mention", "negotiate", "note",
      "observe", "offer", "oppose", "order", "persuade", "pledge", "point", "point out",
      "praise", "pray", "predict", "prefer", "present", "promise", "prompt", "propose",
      "protest", "prove", "provoke", "question", "quote", "raise", "rally", "read",
      "reaffirm", "realise", "realize", "rebut", "recall", "reckon", "recommend", "refer",
      "reflect", "refuse", "refute", "reiterate", "reject", "relate", "remark",
      "remember", "remind", "repeat", "reply", "report", "request", "respond",
      "restate", "reveal", "rule", "say", "see", "show", "signal", "sing",
      "slam", "speculate", "spoke", "spread", "state", "stipulate", "stress",
      "suggest", "support", "suppose", "surmise", "suspect", "swear", "teach",
      "tell", "testify", "think", "threaten", "told", "uncover", "underline",
      "underscore", "urge", "voice", "vow", "warn", "welcome",
      "wish", "wonder", "worry", "write"));
  public final Set<String> reportNoun = new HashSet<String>(Arrays.asList(
      "ABCs", "acclamation", "account", "accusation", "acknowledgment", "address", "addressing",
      "admission", "advertisement", "advice", "advisory", "affidavit", "affirmation", "alert",
      "allegation", "analysis", "anecdote", "annotation", "announcement", "answer", "antiphon",
      "apology", "applause", "appreciation", "argument", "arraignment", "article", "articulation",
      "aside", "assertion", "asseveration", "assurance", "attestation", "attitude", "audience",
      "averment", "avouchment", "avowal", "axiom", "backcap", "band-aid", "basic", "belief", "bestowal",
      "bill", "blame", "blow-by-blow", "bomb", "book", "bow", "break", "breakdown", "brief", "briefing",
      "broadcast", "broadcasting", "bulletin", "buzz", "cable", "calendar", "call", "canard", "canon",
      "card", "cause", "censure", "certification", "characterization", "charge", "chat", "chatter",
      "chitchat", "chronicle", "chronology", "citation", "claim", "clarification", "close", "cognizance",
      "comeback", "comment", "commentary", "communication", "communique", "composition", "concept",
      "concession", "conference", "confession", "confirmation", "conjecture", "connotation", "construal",
      "construction", "consultation", "contention", "contract", "convention", "conversation", "converse",
      "conviction", "cooler", "copy", "counterclaim", "crack", "credenda", "credit", "creed", "critique",
      "cry", "data", "declaration", "defense", "definition", "delineation", "delivery", "demonstration",
      "denial", "denotation", "depiction", "deposition", "description", "detail", "details", "detention",
      "dialogue", "diction", "dictum", "digest", "directive", "dirt", "disclosure", "discourse", "discovery",
      "discussion", "dispatch", "display", "disquisition", "dissemination", "dissertation", "divulgence",
      "dogma", "earful", "echo", "edict", "editorial", "ejaculation", "elucidation", "emphasis", "enlightenment",
      "enucleation", "enunciation", "essay", "evidence", "examination", "example", "excerpt", "exclamation",
      "excuse", "execution", "exegesis", "explanation", "explication", "exposing", "exposition", "expounding",
      "expression", "eye-opener", "feedback", "fiction", "findings", "fingerprint", "flash", "formulation",
      "fundamental", "gift", "gloss", "goods", "gospel", "gossip", "grapevine", "gratitude", "greeting",
      "guarantee", "guff", "hail", "hailing", "handout", "hash", "headlines", "hearing", "hearsay", "history",
      "ideas", "idiom", "illustration", "impeachment", "implantation", "implication", "imputation",
      "incrimination", "inculcation", "indication", "indoctrination", "inference", "info", "information",
      "innuendo", "insinuation", "insistence", "instruction", "intelligence", "interpretation", "interview",
      "intimation", "intonation", "issue", "item", "itemization", "justification", "key", "knowledge",
      "language", "leak", "letter", "line", "lip", "list", "locution", "lowdown", "make", "manifesto",
      "meaning", "meeting", "mention", "message", "missive", "mitigation", "monograph", "motive", "murmur",
      "narration", "narrative", "news", "nod", "note", "notice", "notification", "oath", "observation",
      "okay", "opinion", "oral", "outline", "paper", "parley", "particularization", "phrase", "phraseology",
      "phrasing", "picture", "piece", "pipeline", "pitch", "plea", "plot", "poop", "portraiture", "portrayal",
      "position", "potboiler", "prating", "precept", "prediction", "presentation", "presentment", "principle",
      "proclamation", "profession", "program", "promulgation", "pronouncement", "pronunciation", "propaganda",
      "prophecy", "proposal", "proposition", "prosecution", "protestation", "publication", "publicity",
      "publishing", "quotation", "ratification", "reaction", "reason", "rebuttal", "receipt", "recital",
      "recitation", "recognition", "record", "recount", "recountal", "refutation", "regulation", "rehearsal",
      "rejoinder", "relation", "release", "remark", "rendition", "repartee", "reply", "report", "reporting",
      "representation", "resolution", "response", "result", "retort", "return", "revelation", "review",
      "riposte", "rule", "rumble", "rumor", "rundown", "salutation", "salute", "saying", "scandal", "scoop",
      "scuttlebutt", "sense", "showing", "sign", "signature", "significance", "sketch", "skinny", "solution",
      "speaking", "specification", "speech", "spiel", "statement", "story", "study", "style", "suggestion",
      "summarization", "summary", "summons", "tale", "talk", "talking", "tattle", "teaching", "telecast",
      "telegram", "telling", "tenet", "term", "testimonial", "testimony", "text", "theme", "thesis", "tidings",
      "topper", "tract", "tractate", "tradition", "translation", "treatise", "utterance", "vent", "ventilation",
      "verbalization", "version", "vignette", "vindication", "vocalization", "voice", "voicing", "warning",
      "warrant", "whispering", "wire", "wisecrack", "word", "work", "writ", "write-up", "writeup", "writing", "yarn"
  ));
  public final Set<String> nonEventVerb = new HashSet<String>(Arrays.asList(
      "be", "have", "seem"));

  public final Set<String> nonWords = new HashSet<String>(Arrays.asList("mm", "hmm", "ahem", "um"));
  public final Set<String> copulas = new HashSet<String>(Arrays.asList("is","are","were", "was","be", "been","become","became","becomes","seem","seemed","seems","remain","remains","remained"));
  public final Set<String> quantifiers = new HashSet<String>(Arrays.asList("not","every","any","none","everything","anything","nothing","all","enough"));
  public final Set<String> parts = new HashSet<String>(Arrays.asList("half","one","two","three","four","five","six","seven","eight","nine","ten","hundred","thousand","million","billion","tens","dozens","hundreds","thousands","millions","billions","group","groups","bunch","number","numbers","pinch","amount","amount","total","all","mile","miles","pounds"));
  public final Set<String> temporals = new HashSet<String>(Arrays.asList(
      "second", "minute", "hour", "day", "week", "month", "year", "decade", "century", "millennium",
      "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "now",
      "yesterday", "tomorrow", "age", "time", "era", "epoch", "morning", "evening", "day", "night", "noon", "afternoon",
      "semester", "trimester", "quarter", "term", "winter", "spring", "summer", "fall", "autumn", "season",
      "january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"));


  public final Set<String> femalePronouns = new HashSet<String>(Arrays.asList(new String[]{ "her", "hers", "herself", "she" }));
  public final Set<String> malePronouns = new HashSet<String>(Arrays.asList(new String[]{ "he", "him", "himself", "his" }));
  public final Set<String> neutralPronouns = new HashSet<String>(Arrays.asList(new String[]{ "it", "its", "itself", "where", "here", "there", "which" }));
  public final Set<String> possessivePronouns = new HashSet<String>(Arrays.asList(new String[]{ "my", "your", "his", "her", "its","our","their","whose" }));
  public final Set<String> otherPronouns = new HashSet<String>(Arrays.asList(new String[]{ "who", "whom", "whose", "where", "when","which" }));
  public final Set<String> thirdPersonPronouns = new HashSet<String>(Arrays.asList(new String[]{ "he", "him", "himself", "his", "she", "her", "herself", "hers", "her", "it", "itself", "its", "one", "oneself", "one's", "they", "them", "themself", "themselves", "theirs", "their", "they", "them", "'em", "themselves" }));
  public final Set<String> secondPersonPronouns = new HashSet<String>(Arrays.asList(new String[]{ "you", "yourself", "yours", "your", "yourselves" }));
  public final Set<String> firstPersonPronouns = new HashSet<String>(Arrays.asList(new String[]{ "i", "me", "myself", "mine", "my", "we", "us", "ourself", "ourselves", "ours", "our" }));
  public final Set<String> moneyPercentNumberPronouns = new HashSet<String>(Arrays.asList(new String[]{ "it", "its" }));
  public final Set<String> dateTimePronouns = new HashSet<String>(Arrays.asList(new String[]{ "when" }));
  public final Set<String> organizationPronouns = new HashSet<String>(Arrays.asList(new String[]{ "it", "its", "they", "their", "them", "which"}));
  public final Set<String> locationPronouns = new HashSet<String>(Arrays.asList(new String[]{ "it", "its", "where", "here", "there" }));
  public final Set<String> inanimatePronouns = new HashSet<String>(Arrays.asList(new String[]{ "it", "itself", "its", "where", "when" }));
  public final Set<String> animatePronouns = new HashSet<String>(Arrays.asList(new String[]{ "i", "me", "myself", "mine", "my", "we", "us", "ourself", "ourselves", "ours", "our", "you", "yourself", "yours", "your", "yourselves", "he", "him", "himself", "his", "she", "her", "herself", "hers", "her", "one", "oneself", "one's", "they", "them", "themself", "themselves", "theirs", "their", "they", "them", "'em", "themselves", "who", "whom", "whose" }));
  public final Set<String> indefinitePronouns = new HashSet<String>(Arrays.asList(new String[]{"another", "anybody", "anyone", "anything", "each", "either", "enough", "everybody", "everyone", "everything", "less", "little", "much", "neither", "no one", "nobody", "nothing", "one", "other", "plenty", "somebody", "someone", "something", "both", "few", "fewer", "many", "others", "several", "all", "any", "more", "most", "none", "some", "such"}));
  public final Set<String> relativePronouns = new HashSet<String>(Arrays.asList(new String[]{"that","who","which","whom","where","whose"}));
  public final Set<String> GPEPronouns = new HashSet<String>(Arrays.asList(new String[]{ "it", "itself", "its", "they","where" }));
  public final Set<String> pluralPronouns = new HashSet<String>(Arrays.asList(new String[]{ "we", "us", "ourself", "ourselves", "ours", "our", "yourself", "yourselves", "they", "them", "themself", "themselves", "theirs", "their" }));
  public final Set<String> singularPronouns = new HashSet<String>(Arrays.asList(new String[]{ "i", "me", "myself", "mine", "my", "yourself", "he", "him", "himself", "his", "she", "her", "herself", "hers", "her", "it", "itself", "its", "one", "oneself", "one's" }));
  public final Set<String> facilityVehicleWeaponPronouns = new HashSet<String>(Arrays.asList(new String[]{ "it", "itself", "its", "they", "where" }));
  public final Set<String> miscPronouns = new HashSet<String>(Arrays.asList(new String[]{"it", "itself", "its", "they", "where" }));
  public final Set<String> reflexivePronouns = new HashSet<String>(Arrays.asList(new String[]{"myself", "yourself", "yourselves", "himself", "herself", "itself", "ourselves", "themselves", "oneself"}));
  public final Set<String> transparentNouns = new HashSet<String>(Arrays.asList(new String[]{"bunch", "group",
      "breed", "class", "ilk", "kind", "half", "segment", "top", "bottom", "glass", "bottle",
      "box", "cup", "gem", "idiot", "unit", "part", "stage", "name", "division", "label", "group", "figure",
      "series", "member", "members", "first", "version", "site", "side", "role", "largest", "title", "fourth",
      "third", "second", "number", "place", "trio", "two", "one", "longest", "highest", "shortest",
      "head", "resident", "collection", "result", "last"
  }));
  public final Set<String> stopWords = new HashSet<String>(Arrays.asList(new String[]{"a", "an", "the", "of", "at",
      "on", "upon", "in", "to", "from", "out", "as", "so", "such", "or", "and", "those", "this", "these", "that",
      "for", ",", "is", "was", "am", "are", "'s", "been", "were"}));

  public final Set<String> notOrganizationPRP = new HashSet<String>(Arrays.asList(new String[]{"i", "me", "myself",
      "mine", "my", "yourself", "he", "him", "himself", "his", "she", "her", "herself", "hers", "here"}));

  public final Set<String> personPronouns = new HashSet<String>();
  public final Set<String> allPronouns = new HashSet<String>();

  public final Map<String, String> statesAbbreviation = new HashMap<String, String>();
  public final Map<String, Set<String>> demonyms = new HashMap<String, Set<String>>();
  public final Set<String> demonymSet = new HashSet<String>();
  public final Set<String> adjectiveNation = new HashSet<String>();

  public final Set<String> countries = new HashSet<String>();
  public final Set<String> statesAndProvinces = new HashSet<String>();

  public final Set<String> neutralWords = new HashSet<String>();
  public final Set<String> femaleWords = new HashSet<String>();
  public final Set<String> maleWords = new HashSet<String>();

  public final Set<String> pluralWords = new HashSet<String>();
  public final Set<String> singularWords = new HashSet<String>();

  public final Set<String> inanimateWords = new HashSet<String>();
  public final Set<String> animateWords = new HashSet<String>();

  public final Map<List<String>, int[]> genderNumber = new HashMap<List<String>, int[]>();

  public final Map<String, Integer> schemaID = new HashMap<String, Integer>();
  
  public final TfIdf tfIdf = new TfIdf();

  public final Map<String, Set<String>> thesaurusVerb = new HashMap<String, Set<String>>();
  public final Map<String, Set<String>> thesaurusNoun = new HashMap<String, Set<String>>();
  public final Map<String, Set<String>> thesaurusAdj = new HashMap<String, Set<String>>();
  
  public final ArrayList<Counter<Pair<String, String>>> corefDictionary = new ArrayList<Counter<Pair<String, String>>>(4);
  public final Counter<Pair<String, String>> corefDictionaryNPMI = new ClassicCounter<Pair<String, String>>();
  public final HashMap<String,Counter<String>> signatures = new HashMap<String,Counter<String>>();

  private void setPronouns() {
    for(String s: animatePronouns){
      personPronouns.add(s);
    }

    allPronouns.addAll(firstPersonPronouns);
    allPronouns.addAll(secondPersonPronouns);
    allPronouns.addAll(thirdPersonPronouns);
    allPronouns.addAll(otherPronouns);

    stopWords.addAll(allPronouns);
  }

  public void loadStateAbbreviation(String statesFile) {
    BufferedReader reader = null;
    try {
      reader = new BufferedReader(new InputStreamReader(IOUtils.getInputStreamFromURLOrClasspathOrFileSystem(statesFile)));
      while(reader.ready()){
        String[] tokens = reader.readLine().split("\t");
        statesAbbreviation.put(tokens[1], tokens[0]);
        statesAbbreviation.put(tokens[2], tokens[0]);
      }
    } catch (IOException e){
      throw new RuntimeIOException(e);
    } finally {
      IOUtils.closeIgnoringExceptions(reader);
    }
  }

  private void loadDemonymLists(String demonymFile) {
    BufferedReader reader = null;
    try {
      reader = new BufferedReader(new InputStreamReader(IOUtils.getInputStreamFromURLOrClasspathOrFileSystem(demonymFile)));
      while(reader.ready()){
        String[] line = reader.readLine().split("\t");
        if(line[0].startsWith("#")) continue;
        Set<String> set = new HashSet<String>();
        for(String s : line){
          set.add(s.toLowerCase());
          demonymSet.add(s.toLowerCase());
        }
        demonyms.put(line[0].toLowerCase(), set);
      }
      adjectiveNation.addAll(demonymSet);
      adjectiveNation.removeAll(demonyms.keySet());
    } catch (IOException e){
      throw new RuntimeIOException(e);
    } finally {
      IOUtils.closeIgnoringExceptions(reader);
    }
  }

  private static void getWordsFromFile(String filename, Set<String> resultSet, boolean lowercase) throws IOException {
    BufferedReader reader = new BufferedReader(new InputStreamReader(IOUtils.getInputStreamFromURLOrClasspathOrFileSystem(filename)));
    while(reader.ready()) {
      if(lowercase) resultSet.add(reader.readLine().toLowerCase());
      else resultSet.add(reader.readLine());
    }
    IOUtils.closeIgnoringExceptions(reader);
  }

  private void loadAnimacyLists(String animateWordsFile, String inanimateWordsFile) {
    try {
      getWordsFromFile(animateWordsFile, animateWords, false);
      getWordsFromFile(inanimateWordsFile, inanimateWords, false);
    } catch (IOException e) {
      throw new RuntimeIOException(e);
    }
  }

  private void loadGenderLists(String maleWordsFile, String neutralWordsFile, String femaleWordsFile) {
    try {
      getWordsFromFile(maleWordsFile, maleWords, false);
      getWordsFromFile(neutralWordsFile, neutralWords, false);
      getWordsFromFile(femaleWordsFile, femaleWords, false);
    } catch (IOException e) {
      throw new RuntimeIOException(e);
    }
  }

  private void loadNumberLists(String pluralWordsFile, String singularWordsFile) {
    try {
      getWordsFromFile(pluralWordsFile, pluralWords, false);
      getWordsFromFile(singularWordsFile, singularWords, false);
    } catch (IOException e) {
      throw new RuntimeIOException(e);
    }
  }
  private void loadStatesLists(String file) {
    try {
      getWordsFromFile(file, statesAndProvinces, true);
    } catch (IOException e) {
      throw new RuntimeIOException(e);
    }
  }

  private void loadCountriesLists(String file) {
    try{
      BufferedReader reader = new BufferedReader(new InputStreamReader(IOUtils.getInputStreamFromURLOrClasspathOrFileSystem(file)));
      while(reader.ready()) {
        String line = reader.readLine();
        countries.add(line.split("\t")[1].toLowerCase());
      }
      reader.close();
    } catch (IOException e) {
      throw new RuntimeIOException(e);
    }
  }

  private void loadGenderNumber(String file){
    try {
      BufferedReader reader = new BufferedReader(new InputStreamReader(IOUtils.getInputStreamFromURLOrClasspathOrFileSystem(file)));
      String line;
      while ((line = reader.readLine())!=null){
        String[] split = line.split("\t");
        List<String> tokens = new ArrayList<String>(Arrays.asList(split[0].split(" ")));
        String[] countStr = split[1].split(" ");
        int[] counts = new int[4];
        counts[0] = Integer.parseInt(countStr[0]);
        counts[1] = Integer.parseInt(countStr[1]);
        counts[2] = Integer.parseInt(countStr[2]);
        counts[3] = Integer.parseInt(countStr[3]);

        genderNumber.put(tokens, counts);
      }
      reader.close();
    } catch (IOException e) {
      throw new RuntimeIOException(e);
    }
  }
  private void loadExtraGender(String file){
    BufferedReader reader = null;
    try {
      reader = new BufferedReader(new InputStreamReader(IOUtils.getInputStreamFromURLOrClasspathOrFileSystem(file)));
      while(reader.ready()) {
        String[] split = reader.readLine().split("\t");
        if(split[1].equals("MALE")) maleWords.add(split[0]);
        else if(split[1].equals("FEMALE")) femaleWords.add(split[0]);
      }
    } catch (IOException e){
      throw new RuntimeIOException(e);
    } finally {
      IOUtils.closeIgnoringExceptions(reader);
    }
  }

  private void loadSchemas(String file) {
    BufferedReader reader = null;
    try {
      reader = new BufferedReader(new InputStreamReader(IOUtils.getInputStreamFromURLOrClasspathOrFileSystem(file)));
      int id = 1;
      while(reader.ready()) {
        String[] split = reader.readLine().split("\\s+");
        for(String verb : split) {
          schemaID.put(verb, id);
        }
        id++;
      }
    } catch (IOException e) {
      throw new RuntimeException(e);
    } finally {
      IOUtils.closeIgnoringExceptions(reader);
    }
  }

  /** load Dekang Lin's thesaurus (Proximity based). */
  private void loadThesaurus(String file, Map<String, Set<String>> thesaurus) {
    BufferedReader reader = null;
    try {
      reader = new BufferedReader(new InputStreamReader(IOUtils.getInputStreamFromURLOrClasspathOrFileSystem(file)));
      int count = -1;
      String word = "";
      while(reader.ready()) {
        String[] split = reader.readLine().toLowerCase().split("\t");
        if(split[0].startsWith("(")) {
          word = split[0].split("\\(")[1].trim();
          if(word.startsWith("C_")) word = word.substring(2);
          thesaurus.put(word, new HashSet<String>());
          count = 0;
        } else if(split[0].equals("))")) {
          continue;
        } else {
          if(count++ < THESAURUS_THRESHOLD) {
            double score = Double.parseDouble(split[1]);
            if(split[0].startsWith("C_")) split[0] = split[0].substring(2);
            if(score > THESAURUS_SCORE_THRESHOLD) thesaurus.get(word).add(split[0]);
          }
        }
      }
    } catch (IOException e) {
      throw new RuntimeException(e);
    } finally {
      IOUtils.closeIgnoringExceptions(reader);
    }
  }
  
  /** load the coref dictionary created by Marta and Matt */
  private void loadCorefDictionary(String file, 
      ArrayList<Counter<Pair<String, String>>> dict) {  

    for(int i = 0; i < 4; i++) dict.add(new ClassicCounter<Pair<String, String>>());

    BufferedReader reader = null;
    try {
      reader = new BufferedReader(new InputStreamReader(IOUtils.getInputStreamFromURLOrClasspathOrFileSystem(file)));

      while(reader.ready()) {       
        String[] split = reader.readLine().toLowerCase().split("\t");
        ArrayList<Pair<String, String>> pair_cols = new ArrayList<Pair<String, String>>(4);
        pair_cols.add(new Pair<String, String>(split[1], split[6]));
        pair_cols.add(new Pair<String, String>(split[2], split[7]));
        pair_cols.add(new Pair<String, String>(split[3], split[8]));
        pair_cols.add(new Pair<String, String>(split[4], split[9]));

        for(int i = 0; i < 4; i++){
          Pair<String, String> reversed_pair = 
              new Pair<String, String>(pair_cols.get(i).second(), pair_cols.get(i).first());    
          if(dict.get(i).containsKey(reversed_pair)){
            dict.get(i).incrementCount(reversed_pair);
          } else {
            dict.get(i).incrementCount(pair_cols.get(i));
          }          
        }
      }     
    } catch (IOException e) {
      throw new RuntimeException(e);
    } finally {
      IOUtils.closeIgnoringExceptions(reader);
    }
  }
  
  private void loadCorefDictionaryNPMI(String file, Counter<Pair<String, String>> dict) {
    BufferedReader reader = null;
    try {
      reader = new BufferedReader(new InputStreamReader(IOUtils.getInputStreamFromURLOrClasspathOrFileSystem(file)));

      while(reader.ready()) {
        String[] split = reader.readLine().split("\t");
        dict.setCount(new Pair<String, String>(split[0], split[1]), Double.parseDouble(split[2]));      
      }
    } catch (IOException e) {
      throw new RuntimeException(e);
    } finally {
      IOUtils.closeIgnoringExceptions(reader);
    }
  }

  private void loadSignatures(String file, HashMap<String,Counter<String>> sigs) {
    BufferedReader reader = null;
    try {
      reader = new BufferedReader(new InputStreamReader(IOUtils.getInputStreamFromURLOrClasspathOrFileSystem(file)));

      while(reader.ready()) {       
        String[] split = reader.readLine().split("\t");
        Counter<String> cntr = new ClassicCounter<String>();
        sigs.put(split[0], cntr);
        for (int i = 1; i < split.length; i=i+2) {
          cntr.setCount(split[i], Double.parseDouble(split[i+1]));
        }
      }     
    } catch (IOException e) {
      throw new RuntimeException(e);
    } finally {
      IOUtils.closeIgnoringExceptions(reader);
    }
  }

  public static void main(String[] args) {
    Dictionaries d = new Dictionaries();

  }

  public Dictionaries(Properties props) {
    this(props.getProperty(Constants.DEMONYM_PROP, DefaultPaths.DEFAULT_DCOREF_DEMONYM),
        props.getProperty(Constants.ANIMATE_PROP, DefaultPaths.DEFAULT_DCOREF_ANIMATE),
        props.getProperty(Constants.INANIMATE_PROP, DefaultPaths.DEFAULT_DCOREF_INANIMATE),
        props.getProperty(Constants.MALE_PROP, DefaultPaths.DEFAULT_DCOREF_MALE),
        props.getProperty(Constants.NEUTRAL_PROP, DefaultPaths.DEFAULT_DCOREF_NEUTRAL),
        props.getProperty(Constants.FEMALE_PROP, DefaultPaths.DEFAULT_DCOREF_FEMALE),
        props.getProperty(Constants.PLURAL_PROP, DefaultPaths.DEFAULT_DCOREF_PLURAL),
        props.getProperty(Constants.SINGULAR_PROP, DefaultPaths.DEFAULT_DCOREF_SINGULAR),
        props.getProperty(Constants.STATES_PROP, DefaultPaths.DEFAULT_DCOREF_STATES),
        props.getProperty(Constants.GENDER_NUMBER_PROP, DefaultPaths.DEFAULT_DCOREF_GENDER_NUMBER),
        props.getProperty(Constants.COUNTRIES_PROP, DefaultPaths.DEFAULT_DCOREF_COUNTRIES),
        props.getProperty(Constants.STATES_PROVINCES_PROP, DefaultPaths.DEFAULT_DCOREF_STATES_AND_PROVINCES),
        props.getProperty(Constants.EXTRA_GENDER_PROP, DefaultPaths.DEFAULT_DCOREF_EXTRA_GENDER),
        props.getProperty(Constants.SCHEMAS_PROP, "/scr/heeyoung/coref/jcoref/baseline/schemas.txt"),  // temporary to avoid change core
        props.getProperty(Constants.THESAURUS_VERB_PROP, "/scr/heeyoung/corpus/DekangLinSyntaxBasedThesaurus/simV.lsp"),
        props.getProperty(Constants.THESAURUS_NOUN_PROP, "/scr/heeyoung/corpus/DekangLinSyntaxBasedThesaurus/simN.lsp"),
        props.getProperty(Constants.THESAURUS_ADJ_PROP, "/scr/heeyoung/corpus/DekangLinSyntaxBasedThesaurus/simA.lsp"),
        props.getProperty(Constants.COREF_DICT, "/user/recasens/data/corefdict/dict-patterns-techmeme"),
        props.getProperty(Constants.COREF_DICT_NPMI, "/user/recasens/data/corefdict/col1-npmi"),
        props.getProperty(Constants.CONTEXT_SIGNATURES, "/user/mattcan/scr/signatures.txt"),
        Boolean.parseBoolean(props.getProperty(Constants.BIG_GENDER_NUMBER_PROP, "false")) ||
        Boolean.parseBoolean(props.getProperty(Constants.REPLICATECONLL_PROP, "false")));
  }

  public Dictionaries(
      String demonymWords,
      String animateWords,
      String inanimateWords,
      String maleWords,
      String neutralWords,
      String femaleWords,
      String pluralWords,
      String singularWords,
      String statesWords,
      String genderNumber,
      String countries,
      String states,
      String extraGender,
      String schemas,
      String thesaurusVerbFile,
      String thesaurusNounFile,
      String thesaurusAdjFile,
      String corefDict,
      String corefDictNPMI,
      String signaturesFile,
      boolean loadBigGenderNumber) {
    loadDemonymLists(demonymWords);
    loadStateAbbreviation(statesWords);
    if(Constants.USE_ANIMACY_LIST) loadAnimacyLists(animateWords, inanimateWords);
    if(Constants.USE_GENDER_LIST) loadGenderLists(maleWords, neutralWords, femaleWords);
    if(Constants.USE_NUMBER_LIST) loadNumberLists(pluralWords, singularWords);
    if(loadBigGenderNumber) loadGenderNumber(genderNumber);
    loadCountriesLists(countries);
    loadStatesLists(states);
    loadExtraGender(extraGender);
    loadSchemas(schemas);
    loadThesaurus(thesaurusVerbFile, thesaurusVerb);
    loadThesaurus(thesaurusNounFile, thesaurusNoun);
    loadThesaurus(thesaurusAdjFile, thesaurusAdj);
    //loadCorefDictionary(corefDict, corefDictionary);   //MARTA: Uncomment for coref-dict
    //loadCorefDictionaryNPMI(corefDictNPMI, corefDictionaryNPMI);  //MARTA: Uncomment for coref-dict
    //loadSignatures(signaturesFile, signatures);   //MARTA: Uncomment for coref-dict
    setPronouns();
  }

  public Dictionaries() {
    this(new Properties());
  }
}
