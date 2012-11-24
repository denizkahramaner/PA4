package edu.stanford.nlp.jcoref.dcoref;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import net.didion.jwnl.data.POS;
import net.didion.jwnl.data.Synset;
import edu.stanford.nlp.jcoref.dcoref.Dictionaries.MentionType;
import edu.stanford.nlp.ling.CoreAnnotations.LemmaAnnotation;
import edu.stanford.nlp.util.Pair;

/**
 * Semantic knowledge based on WordNet
 */
public class WordNet implements Serializable {
  private static final long serialVersionUID = 7112442030504673315L;
  HashMap<Pair<Integer, Integer>, Double> wnSimilarityScore;
  final WordNetHelper wn;
  private final int maxNodeDistance = 2;
  private static final String WN_CONFIG_FILE = "/u/nlp/data/wordnet/jwnl/jwnl_map_properties.xml";
  public Map<Synset, Integer> synsetClusters;

  public WordNet () {
    System.err.println("WordNet.java: load " + WN_CONFIG_FILE + " for WordNetHelper");
    wn = new WordNetHelper(WN_CONFIG_FILE);
    synsetClusters = WordNetHelper.synsetClustering();
  }

  public boolean alias(Mention mention, Mention antecedent) {
    return checkSynonym(mention, antecedent);
  }

  public static class WNsynset implements Serializable {
    private static final long serialVersionUID = 1623663405576312167L;
    public Synset[] synsets;
    public Synset[] nominalizedSynsets;
    public String word;
    protected WNsynset(Synset[] synsets, Synset[] nominalized, String w) {
      this.synsets = synsets;
      this.nominalizedSynsets = nominalized;
      this.word = w;
    }
  }

  public boolean checkSynonym(Mention m, Mention ant) {
    // Proper <- Proper or Nominal <- Nominal only
    if(m.mentionType!=ant.mentionType || ant.isPronominal()) return false;

    // sentence distance constraints
    //    if(Math.abs(m.sentNum-ant.sentNum) > 3 && m.mentionType==MentionType.NOMINAL) return false;

    if(m.synsets==null || ant.synsets==null) return false;
    if(m.synsets.word.equalsIgnoreCase(ant.synsets.word)) return false;   // to remove same head?

    Set<String> stringSet = new HashSet<String>();
    stringSet.addAll(Arrays.asList(m.synsets.word.toLowerCase().split("_")));
    stringSet.addAll(Arrays.asList(ant.synsets.word.toLowerCase().split("_")));

    // return false if both have proper noun which is not involved in semantic match
    if(Rules.entityHaveExtraProperNoun(m, ant, stringSet)) return false;

    return checkSynonym(m.synsets, ant.synsets);
  }

  protected boolean checkSynonym(WNsynset mSynsets, WNsynset antSynsets) {
    if(mSynsets==null || antSynsets == null) return false;
    Synset abstractEntity = wn.synsetsOf("abstract_entity", POS.NOUN)[0];
    Synset organization = wn.synsetsOf("organization", POS.NOUN)[0];
    for (Synset mSyn : mSynsets.synsets) {
      for (Synset aSyn : antSynsets.synsets) {
        List<Synset> hypernyms = wn.hypernymChain(mSyn);
        if(hypernyms!=null && hypernyms.contains(abstractEntity)
            && !hypernyms.contains(organization)) continue;
        if(mSyn.equals(aSyn)) return true;
      }
    }
    return false;
  }


  public boolean checkHypernym(Mention m, Mention ant) {
    // Proper <- Nominal or Nominal <- Nominal only
    if(m.mentionType!=MentionType.NOMINAL || ant.isPronominal()) return false;

    // sentence distance constraints
    //    if(Math.abs(m.sentNum-ant.sentNum) > 3) return false;

    if(m.synsets==null || ant.synsets==null) return false;
    if(m.synsets.word.equalsIgnoreCase(ant.synsets.word)) return false;

    Set<String> stringSet = new HashSet<String>();
    stringSet.addAll(Arrays.asList(m.synsets.word.toLowerCase().split("_")));
    stringSet.addAll(Arrays.asList(ant.synsets.word.toLowerCase().split("_")));

    // return false if both have proper noun which is not involved in semantic match
    if(Rules.entityHaveExtraProperNoun(m, ant, stringSet)) return false;

    return checkHypernym(m.synsets, ant.synsets);
  }
  protected boolean checkHypernym(WNsynset mSynsets, WNsynset antSynsets) {
    if(mSynsets==null || antSynsets == null) return false;
    HashSet<Synset> antHypernyms = getNonAbstractHypernyms(antSynsets);
    // synonym or later mention should be more general
    for (Synset s : mSynsets.synsets){
      if(antHypernyms.contains(s)) return true;
    }
    return false;
  }

  protected HashSet<Synset> getNonAbstractHypernyms(WNsynset synsets) {
    return getNonAbstractHypernyms(synsets, maxNodeDistance);
  }

  protected HashSet<Synset> getNonAbstractHypernyms(WNsynset synsets, int depth) {
    if (synsets==null) return new HashSet<Synset>();
    Synset abstractEntity = wn.synsetsOf("abstract_entity", POS.NOUN)[0];
    Synset organization = wn.synsetsOf("organization", POS.NOUN)[0];

    HashSet<Synset> hypernyms = new HashSet<Synset>();
    for (Synset s : synsets.synsets){
      List<Synset> hypernymChain = wn.hypernymChain(s);
      if(hypernymChain!=null && hypernymChain.contains(abstractEntity)
          && !hypernymChain.contains(organization)) continue;
      if(wn.hypernymChain(s, depth)!=null) {
        hypernyms.addAll(wn.hypernymChain(s, depth));
      }
    }
    return hypernyms;
  }

  protected WNsynset findSynset(String term) {
    if(term.split(" ").length > 10) return null; // too long
    Synset[] synsets = wn.synsetsOf(term, POS.NOUN);
    if(synsets==null) return null;

    List<String> words = wn.wordsInSynset(synsets[0]);

    // make sure search term didn't changed
    // keep the original word
    for (String s : words) {
      if(s.equalsIgnoreCase(term.replace(' ', '_'))) {
        WNsynset syn = new WNsynset(synsets, synsets, s);
        return syn;
      }
    }
    return null;
  }

  protected WNsynset findSynset(List<String> terms) {
    for(String term : terms) {
      if (term.contains("://")) {
        //        logger.warning("Skip term " + term);
        continue;  // urls
      }
      if(term.split(" ").length > 10) {
        //        logger.info("Skip term " + term);
        continue; // too long
      }
      //      logger.info("Lookup term " + term);
      Synset[] synsets = wn.synsetsOf(term, POS.NOUN);
      if(synsets==null) continue;

      List<String> words = wn.wordsInSynset(synsets[0]);

      // make sure search term didn't changed
      // keep the original word
      for (String s : words) {
        if(s.equalsIgnoreCase(term.replace(' ', '_'))) {
          WNsynset syn = new WNsynset(synsets, synsets, s);
          return syn;
        }
      }
    }
    return null;
  }

  protected Set<String> wordnetCategory(String query){
    Set<String> ners = new HashSet<String>();
    Synset person = wn.synsetsOf("person", POS.NOUN)[0];
    Synset people = wn.synsetsOf("people", POS.NOUN)[0];
    Synset socialGroup = wn.synsetsOf("social group", POS.NOUN)[0];
    Synset organization = wn.synsetsOf("organization", POS.NOUN)[0];
    Synset location = wn.synsetsOf("location", POS.NOUN)[0];
    WNsynset synset = findSynset(new ArrayList<String>(Arrays.asList(new String[]{query})));
    Set<Synset> hypernyms = new HashSet<Synset>();
    if(synset!=null) hypernyms = getNonAbstractHypernyms(synset);
    else ners.add("O");
    if(hypernyms.contains(person)) ners.add("PERSON");
    if(hypernyms.contains(organization) || hypernyms.contains(socialGroup)) ners.add("ORGANIZATION");
    if(hypernyms.contains(location) || hypernyms.contains(people)) ners.add("LOCATION");
    return ners;
  }

  protected Set<String> getSynonyms(String text) {
    Set<String> synonyms = new HashSet<String>();
    synonyms.add(text);
    WNsynset s = findSynset(text);
    if(s!=null) {
      for(Synset synset : s.synsets){
        for(String synonym : wn.wordsInSynset(synset)) {
          synonyms.add(synonym.replace('_', ' '));
        }
      }
    }
    return synonyms;
  }

  public static boolean similarInWN(CorefCluster mC, CorefCluster aC, double threshold, WordNet wordnet, Dictionaries dict) {
    int total = 0;
    int similarCnt = 0;
    for(Mention m : mC.getCorefMentions()) {
      for(Mention a : aC.getCorefMentions()) {
        total++;
        if(similarInWN(m, a, wordnet, dict)) {
          similarCnt++;
        }
      }
    }
    return ((similarCnt*1.0/total) >= threshold);
  }
  /** check hypernym, synonym, sibling relations in WordNet
   * @param dict */
  public static boolean similarInWN(Mention m, Mention ant, WordNet wordnet, Dictionaries dict) {
    boolean samePOS = (m.isVerb==ant.isVerb);
    if(m.synsets==null || ant.synsets==null) return false;
    Synset[] mSyns = (samePOS || m.synsets.nominalizedSynsets==null)? m.synsets.synsets : m.synsets.nominalizedSynsets;
    Synset[] antSyns = (samePOS || ant.synsets.nominalizedSynsets==null)? ant.synsets.synsets : ant.synsets.nominalizedSynsets;

//    // if a verb is too general, skip
//    if(dict.lightVerb.contains(m.headWord.get(LemmaAnnotation.class))) {
//      return false;
//    }
//    if(dict.lightVerb.contains(ant.headWord.get(LemmaAnnotation.class))) {
//      return false;
//    }

    if(wordnet.checkHypernym(mSyns, antSyns)
        || wordnet.checkSynonym(mSyns, antSyns)
        ){
//        || wordnet.checkSibling(mSyns, antSyns)) {
      return true;
    }
    return false;
  }
  public static boolean synonymInWN(Mention m, Mention ant, WordNet wordnet, Dictionaries dict) {
    boolean samePOS = (m.isVerb==ant.isVerb);
    if(m.synsets==null || ant.synsets==null) return false;
    Synset[] mSyns = (samePOS || m.synsets.nominalizedSynsets==null)? m.synsets.synsets : m.synsets.nominalizedSynsets;
    Synset[] antSyns = (samePOS || ant.synsets.nominalizedSynsets==null)? ant.synsets.synsets : ant.synsets.nominalizedSynsets;
    if(dict.lightVerb.contains(m.headWord.get(LemmaAnnotation.class))
        || dict.lightVerb.contains(ant.headWord.get(LemmaAnnotation.class))) {
      return false;
    }
    if(m.synsets==null || ant.synsets==null) return false;
    if(wordnet.checkSynonym(mSyns, antSyns)) {
      return true;
    }
    return false;
  }
  public static boolean siblingInWN(Mention m, Mention ant, WordNet wordnet, Dictionaries dict) {
    boolean samePOS = (m.isVerb==ant.isVerb);
    if(m.synsets==null || ant.synsets==null) return false;
    Synset[] mSyns = (samePOS || m.synsets.nominalizedSynsets==null)? m.synsets.synsets : m.synsets.nominalizedSynsets;
    Synset[] antSyns = (samePOS || ant.synsets.nominalizedSynsets==null)? ant.synsets.synsets : ant.synsets.nominalizedSynsets;

    // if a verb is too general, skip
    if(dict.lightVerb.contains(m.headWord.get(LemmaAnnotation.class))) {
      return false;
    }
    if(dict.lightVerb.contains(ant.headWord.get(LemmaAnnotation.class))) {
      return false;
    }

    if(wordnet.checkSibling(mSyns, antSyns)) {
      return true;
    }
    return false;
  }
  public static boolean hypernymInWN(Mention m, Mention ant, WordNet wordnet, Dictionaries dict) {
    boolean samePOS = (m.isVerb==ant.isVerb);
    if(m.synsets==null || ant.synsets==null) return false;
    Synset[] mSyns = (samePOS || m.synsets.nominalizedSynsets==null)? m.synsets.synsets : m.synsets.nominalizedSynsets;
    Synset[] antSyns = (samePOS || ant.synsets.nominalizedSynsets==null)? ant.synsets.synsets : ant.synsets.nominalizedSynsets;

    // if a verb is too general, skip
    if(dict.lightVerb.contains(m.headWord.get(LemmaAnnotation.class))) {
      return false;
    }
    if(dict.lightVerb.contains(ant.headWord.get(LemmaAnnotation.class))) {
      return false;
    }

    if(wordnet.checkHypernym(mSyns, antSyns)) {
      return true;
    }
    return false;
  }

  private boolean checkSibling(Synset[] mSyn, Synset[] antSyn) {
    Set<Synset> mHypernyms = new HashSet<Synset>();
    for(Synset mS : mSyn) {
      List<Synset> hypernyms = wn.hypernymChain(mS, 1);
      if(hypernyms!=null) mHypernyms.addAll(hypernyms);
    }
    Set<Synset> aHypernyms = new HashSet<Synset>();
    for(Synset aS : antSyn) {
      List<Synset> hypernyms = wn.hypernymChain(aS, 1);
      if(hypernyms!=null) aHypernyms.addAll(hypernyms);
    }
    mHypernyms.retainAll(aHypernyms);
    if(mHypernyms.size() > 0) {
      return true;
    }
    return false;
  }
  private boolean checkSynonym(Synset[] mSyn, Synset[] antSyn) {
    // use all synsets
    Set<Synset> mSynsets = new HashSet<Synset>(Arrays.asList(mSyn));
    Set<Synset> aSynsets = new HashSet<Synset>(Arrays.asList(antSyn));

    mSynsets.retainAll(aSynsets);
    if(mSynsets.isEmpty()) return false;
    else return true;
  }

  private boolean checkHypernym(Synset[] mSyn, Synset[] antSyn) {
    Set<Synset> mHypernyms = new HashSet<Synset>();
    for(Synset mS : mSyn) {
      List<Synset> hypernyms = wn.hypernymChain(mS, maxNodeDistance);
      if(hypernyms!=null) mHypernyms.addAll(hypernyms);
    }
    Set<Synset> aHypernyms = new HashSet<Synset>();
    for(Synset aS : antSyn) {
      List<Synset> hypernyms = wn.hypernymChain(aS, maxNodeDistance);
      if(hypernyms!=null) aHypernyms.addAll(hypernyms);
    }
    mHypernyms.retainAll(Arrays.asList(antSyn));
    aHypernyms.retainAll(Arrays.asList(mSyn));
    if(mHypernyms.size() > 0 || aHypernyms.size() > 0) {
      return true;
    }
    return false;
  }

  public WNsynset findSynset(String string, POS postag) {
    Synset[] synsets = wn.synsetsOf(string, postag);
    return new WNsynset(synsets, synsets, string);
  }

  public static boolean sameSynsetCluster(Mention m, Mention ant, WordNet wordnet) {
    if(m.synsets==null || ant.synsets==null) return false;
    if(wordnet.synsetClusters.get(m.synsets.synsets[0])
        == wordnet.synsetClusters.get(ant.synsets.synsets[0])) {
      return true;
    }
    return false;
  }

  public static boolean sameSynsetCluster(CorefCluster menCluster, CorefCluster antCluster, WordNet wordnet) {
    for(Mention m : menCluster.getCorefMentions()) {
      for(Mention a : antCluster.getCorefMentions()) {
        if(!sameSynsetCluster(m, a, wordnet)) return false;
      }
    }
    return true;
  }
}
