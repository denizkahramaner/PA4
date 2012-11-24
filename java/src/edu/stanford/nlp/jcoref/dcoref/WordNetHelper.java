package edu.stanford.nlp.jcoref.dcoref;

import java.io.FileInputStream;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;

import net.didion.jwnl.JWNL;
import net.didion.jwnl.JWNLException;
import net.didion.jwnl.data.IndexWord;
import net.didion.jwnl.data.POS;
import net.didion.jwnl.data.Pointer;
import net.didion.jwnl.data.PointerType;
import net.didion.jwnl.data.Synset;
import net.didion.jwnl.data.Word;
import net.didion.jwnl.dictionary.Dictionary;


/** From Nate
 */

/**
 * Helper class to lookup lemma forms in WordNet.
 * Caches lookups in memory to speedup the lookup, but can increase memory size.
 */
public class WordNetHelper implements Serializable {

  private static final long serialVersionUID = -5958349121309174937L;
  private Map<String,String> _verbToLemma;
  private Map<String,String> _nounToLemma;
  private Map<String,String> _adjToLemma;
  private Map<String,Boolean> _isNounEvent;
  private Map<String,Boolean> _isPhysicalObject;
  private Map<String,Boolean> _isMaterial;
  private Map<String,Boolean> _isPersonOrGroup;
  private Map<String,Boolean> _isLocation;
  private Map<String,Boolean> _isStructure;
  private Map<String,Boolean> _isMeasure;

  // HYPERNYM is the main link in WordNet.  However, they also have an "instance hypernym"
  // which does not have a PointerType type in their API, yet appears in their database.
  // It appears with proper names.
  // This is a hack that checks the first character of the PointerType keys:
  //    "@i" as an instance, "@" as the standard hypernym.
  public final char hypernymChar = '@';


  public WordNetHelper(String wordnetPath) {
    // Load WordNet
    try {
      if( wordnetPath.length() > 0 )
        JWNL.initialize(new FileInputStream(wordnetPath));
    } catch( Exception ex ) { ex.printStackTrace(); }
  }

  /**
   * @param word A word
   * @return The lemma of the word if it is a verb, null otherwise
   */
  public String verbToLemma(String word) {
    if( _verbToLemma == null ) _verbToLemma = new HashMap<String, String>();

    // save time with a table lookup
    if( _verbToLemma.containsKey(word) ) return _verbToLemma.get(word);

    try {
      // don't return lemmas for hyphenated words
      if( word.indexOf('-') > -1 || word.indexOf('/') > -1 ) {
        _verbToLemma.put(word, null);
        return null;
      }

      // get the lemma
      IndexWord iword = Dictionary.getInstance().lookupIndexWord(POS.VERB, word);
      if( iword == null ) {
        _verbToLemma.put(word, null);
        return null;
      }
      else {
        String lemma = iword.getLemma();
        if( lemma.indexOf(' ') != -1 ) // Sometimes it returns a two word phrase
          lemma = lemma.trim().replace(' ','_');

        _verbToLemma.put(word, lemma);
        return lemma;
      }
    } catch( Exception ex ) { ex.printStackTrace(); }

    return null;
  }


  /**
   * @param word A word
   * @return The lemma of the word if it is a noun, null otherwise
   */
  public String nounToLemma(String word) {
    if( _nounToLemma == null ) _nounToLemma = new HashMap<String, String>();

    // save time with a table lookup
    if( _nounToLemma.containsKey(word) ) return _nounToLemma.get(word);

    try {
      // don't return lemmas for hyphenated words
      if( word.indexOf('-') > -1 || word.indexOf('/') > -1 ) {
        _nounToLemma.put(word, null);
        return null;
      }

      // get the lemma
      IndexWord iword = Dictionary.getInstance().lookupIndexWord(POS.NOUN, word);
      if( iword == null ) {
        _nounToLemma.put(word, null);
        return null;
      }
      else {
        String lemma = iword.getLemma();


        if( word.equals(lemma) ) {
          // Some nouns have their plural in WordNet as a strange rare word (e.g. devices).
          // Here we guess the single form, and return it if the guess exists (e.g. device).
          if( word.endsWith("es") ) {
            String guess = word.substring(0, word.length()-1);
            IndexWord iGuess = Dictionary.getInstance().lookupIndexWord(POS.NOUN, guess);
            if( iGuess != null && guess.equals(iGuess.getLemma()) ) {
              lemma = guess;
              //              System.out.println("WORDNET guessed singular: " + lemma + " from " + word);
            }
          }

          // "men" and "businessmen" are in WordNet as lemmas ... we need to get the singular man
          else if( word.endsWith("men") ) {
            String guess = word.substring(0, word.length()-2) + "an";
            IndexWord iGuess = Dictionary.getInstance().lookupIndexWord(POS.NOUN, guess);
            if( iGuess != null && guess.equals(iGuess.getLemma()) ) {
              lemma = guess;
              //              System.out.println("WORDNET guessed singular: " + lemma + " from " + word);
            }
          }

          else if( word.equals("people") )
            return "person";
        }

        if( lemma.indexOf(' ') != -1 ) // Sometimes it returns a two word phrase
          lemma = lemma.trim().replace(' ','_');

        _nounToLemma.put(word, lemma);
        return lemma;
      }
    } catch( Exception ex ) { ex.printStackTrace(); }

    return null;
  }


  /**
   * @param word A word
   * @return The lemma of the word if it is an adjective, null otherwise
   */
  public String adjectiveToLemma(String word) {
    if( _adjToLemma == null ) _adjToLemma = new HashMap<String, String>();

    // save time with a table lookup
    if( _adjToLemma.containsKey(word) ) return _adjToLemma.get(word);

    try {
      // don't return lemmas for hyphenated words
      if( word.indexOf('-') > -1 || word.indexOf('/') > -1 ) {
        _adjToLemma.put(word, null);
        return null;
      }

      // get the lemma
      IndexWord iword = Dictionary.getInstance().lookupIndexWord(POS.ADJECTIVE, word);
      if( iword == null ) {
        _adjToLemma.put(word, null);
        return null;
      }
      else {
        String lemma = iword.getLemma();
        if( lemma.indexOf(' ') != -1 ) // Sometimes it returns a two word phrase
          lemma = lemma.trim().replace(' ','_');

        _adjToLemma.put(word, lemma);
        return lemma;
      }
    } catch( Exception ex ) { ex.printStackTrace(); }

    return null;
  }


  /**
   * Uses Treebank tags and calls the correct verb, noun, adj lemmatizer.
   */
  public String lemmatizeTaggedWord(String token, String postag) {
    String lemma = null;

    if( postag.startsWith("VB") )
      lemma = verbToLemma(token);
    else if( postag.startsWith("N") )
      lemma = nounToLemma(token);
    else if( postag.startsWith("J") )
      lemma = adjectiveToLemma(token);
    //    else
    //      System.out.println("Unknown TAG (lemmatize): " + postag);

    if( lemma == null ) lemma = token;

    //    System.out.println("  lemmatizing " + token + " pos " + postag + " = " + lemma);

    return lemma;
  }

  /**
   * @return All synsets for the given word and POS category.
   */
  public Synset[] synsetsOf(String token, POS postag) {
    try {
      IndexWord iword = Dictionary.getInstance().lookupIndexWord(postag, token);
      if( iword != null ) {
        Synset[] synsets = iword.getSenses();
        return synsets;
      }
    } catch( Exception ex ) { ex.printStackTrace(); }
    return null;
  }

  public Synset[] synsetsOfPhrase(String token, POS postag) {
    try {
      token = token.trim().toLowerCase();
      IndexWord iword = Dictionary.getInstance().getIndexWord(postag, token);
      if( iword != null ) {
        Synset[] synsets = iword.getSenses();
        return synsets;
      }
    } catch( Exception ex ) { ex.printStackTrace(); }
    return null;
  }

  /**
   * Returns true if the two tokens are under the same immediate synset (siblings).
   * @param token1 A token.
   * @param token2 A token.
   * @param postag The POS tag of both tokens.
   * @return True if the tokens are siblings, false otherwise.
   */
  public boolean areSiblings(String token1, String token2, POS postag) {
    Synset[] synsets1 = synsetsOf(token1, postag);
    Synset[] synsets2 = synsetsOf(token2, postag);
    if( synsets1 != null && synsets2 != null ) {
      for( int i = 0; i < synsets1.length; i++ ) {
        Synset syn = synsets1[i];
        for( int j = 0; j < synsets2.length; j++ ) {
          if( syn == synsets2[j] )
            return true;
        }
      }
    }
    if( (token1.equals("hurt") && token2.equals("injure")) || (token1.equals("injure") && token2.equals("hurt"))  )
      System.out.println("areSiblings returning false!");
    return false;
  }

  /**
   * @return All lemmas that are under the given synset.
   */
  public List<String> wordsInSynset(Synset synset) {
    List<String> strings = new ArrayList<String>();
    Word[] words = synset.getWords();
    for( Word word : words )
      strings.add(word.getLemma());
    return strings;
  }

  /**
   * Get the chain of parents from the given synset to the top of the
   * wordnet hierarchy.
   */
  public List<Synset> hypernymChain(Synset synset) {
    List<Synset> history = new ArrayList<Synset>();
    history.add(synset);
    return hypernymChain(synset, history);
  }
  /**
   * Get the chain of parents from the given synset to the depth-th parents of the
   * wordnet hierarchy.
   */
  public List<Synset> hypernymChain(Synset synset, int depth) {
    List<Synset> history = new ArrayList<Synset>();
    history.add(synset);
    return hypernymChain(synset, history, depth);
  }

  /**
   * Get the chain of parents from the given synset to the top of the
   * wordnet hierarchy.  Checks for loops (they exist!).
   * @param synset Child synset of which we want the parent chain.
   * @param history The list of synsets we've already traversed.
   * @return The hypernym chain.
   */
  public List<Synset> hypernymChain(Synset synset, List<Synset> history) {
    List<Synset> chain = new ArrayList<Synset>();

    Pointer[] links = synset.getPointers();
    if( links != null ) {
      for( Pointer link : links ) {
        // HYPERNYM is the type of link in WordNet.  However, they also have an "instance hypernym"
        // which does not have a PointerType type in their API, yet appears in their database.
        // This is a hack that checks the key "@i" is an instance, "@" is standard hypernym.
        //        if( link.getType() == PointerType.HYPERNYM ) {
        if( link.getType().getKey().charAt(0) == hypernymChar ) {
          try {
            Synset target = link.getTargetSynset();
            if( !history.contains(target) ) {
              history.add(target);
              chain.add(target);
              List<Synset> upperChain = hypernymChain(target, history);
              if( upperChain != null ) chain.addAll(upperChain);
              // ** There's only one parent per synset, right?
              return chain;
            }
            //else System.out.println("Wordnet found loop at " + synset + "\nhistory=" + history);
          } catch( Exception ex ) { ex.printStackTrace(); }
        }
      }
    }
    return null;
  }

  /**
   * Get the chain of parents from the given synset to the depth-th parents of the
   * wordnet hierarchy.  Checks for loops (they exist!).
   * @param synset Child synset of which we want the parent chain.
   * @param history The list of synsets we've already traversed.
   * @param depth Maximum node distance
   * @return The hypernym chain.
   */
  public List<Synset> hypernymChain(Synset synset, List<Synset> history, int depth) {
    List<Synset> chain = new ArrayList<Synset>();
    if(depth==0) return chain;

    Pointer[] links = synset.getPointers();
    if( links != null ) {
      for( Pointer link : links ) {
        // HYPERNYM is the type of link in WordNet.  However, they also have an "instance hypernym"
        // which does not have a PointerType type in their API, yet appears in their database.
        // This is a hack that checks the key "@i" is an instance, "@" is standard hypernym.
        //        if( link.getType() == PointerType.HYPERNYM ) {
        if( link.getType().getKey().charAt(0) == hypernymChar ) {
          try {
            Synset target = link.getTargetSynset();
            if( !history.contains(target) ) {
              history.add(target);
              chain.add(target);
              List<Synset> upperChain = hypernymChain(target, history, depth-1);
              if( upperChain != null ) chain.addAll(upperChain);
              // ** There's only one parent per synset, right?
              return chain;
            }
            //else System.out.println("Wordnet found loop at " + synset + "\nhistory=" + history);
          } catch( Exception ex ) { ex.printStackTrace(); }
        }
      }
    }
    return null;
  }

  /**
   * Get the chain of parents from the given synset to the top of the
   * wordnet hierarchy.  Return the chain, but rooted with the given
   * synset.
   */
  public List<Synset> hypernymChainKeepChild(Synset synset) {
    List<Synset> chain = new ArrayList<Synset>();
    chain.add(synset);

    List<Synset> parents = hypernymChain(synset);
    if( parents != null )
      chain.addAll(parents);

    return chain;
  }

  /**
   * Get all synsets that are reachable by hypernym relations from this token.
   */
  public Set<Synset> getAllSynsetAncestors(String token, POS tag) {
    Synset[] synsets = synsetsOf(token, tag);
    if( synsets != null ) {
      Set<Synset> allsynsets = new HashSet<Synset>();
      for( Synset synset : synsets ) {
        List<Synset> chain = hypernymChainKeepChild(synset);
        allsynsets.addAll(chain);
      }
      return allsynsets;
    }
    else return null;
  }

  /**
   * Assumes the given token is a noun.
   * @return true if there is some synset with this token that has a
   *              nominalization relation attached to it.
   */
  public boolean isNominalization(String token) {
    Synset[] synsets = synsetsOf(token, POS.NOUN);
    if( synsets == null ) {
      //      System.out.println("isNominalization null synsets: " + token);
    }
    else {
      for( Synset synset : synsets ) {
        Pointer[] links = synset.getPointers();
        if( links != null ) {
          for( Pointer link : links ) {
            // Found a link from this noun as a Nominalization to another.
            if( link.getType() == PointerType.NOMINALIZATION ) {
              // Check that the nominalized word is a verb (e.g. not an adjective).
              try {
                Synset target = link.getTargetSynset();
                if( target.getPOS() == POS.VERB ) {
                  //		  System.out.println("WordNet isNom() link found: " + link);
                  //		  System.out.println(" --> " + link.getTargetSynset());
                  return true;
                }
              } catch( Exception ex ) { ex.printStackTrace(); }
            }
          }
        }
      }
    }
    return false;
  }

  /**
   * Gets all of the verbs that are in the synset of which the given noun token
   * has a nominalization pointer to.
   * @param token A noun e.g. explosion
   * @return A list of strings that are verbs e.g. explode, detonate
   */
  public List<String> getVerbsOfNominalization(String token) {
    Synset[] synsets = synsetsOf(token, POS.NOUN);
    if( synsets != null ) {
      for( Synset synset : synsets ) {
        Pointer[] links = synset.getPointers();
        if( links != null ) {
          for( Pointer link : links ) {
            // Found a link from this noun as a Nominalization to another.
            if( link.getType() == PointerType.NOMINALIZATION ) {
              // Check that the nominalized word is a verb (e.g. not an adjective).
              try {
                Synset target = link.getTargetSynset();
                if( target.getPOS() == POS.VERB ) {
                  Synset verbSynset = link.getTargetSynset();
                  Word[] verbs = verbSynset.getWords();
                  List<String> theverbs = new ArrayList<String>();
                  for( Word verb : verbs )
                    theverbs.add(verb.getLemma());
                  return theverbs;
                }
              } catch( Exception ex ) { ex.printStackTrace(); }
            }
          }
        }
      }
    }
    return null;
  }
  public Synset[] getNominalizationOfVerb(String verb) {
    Synset[] synsets = synsetsOf(verb, POS.VERB);
    List<Synset> nominalizations = new ArrayList<Synset>();
    if( synsets == null ) return null;
    for( Synset synset : synsets ) {
      Pointer[] links = synset.getPointers();
      if( links == null ) continue;
      for( Pointer link : links ) {
        if( link.getType() == PointerType.NOMINALIZATION ) {
          try {
            Synset target = link.getTargetSynset();
            if(target.getPOS() == POS.NOUN) {
              nominalizations.add(target);
            }
          } catch( Exception ex ) {
            throw new RuntimeException(ex);
          }
        }
      }
    }
    Synset[] n = new Synset[nominalizations.size()];
    return nominalizations.toArray(n);
  }


  /**
   * @return True if the token is not known in WordNet
   */
  public boolean isUnknown(String token) {
    Synset[] synsets = synsetsOf(token, POS.NOUN);
    if( synsets == null )
      return true;
    else
      return false;
  }


  public boolean isNounPerson(String token, boolean mainSynsetOnly) {
    return isNounPersonOrGroup(token, mainSynsetOnly, true);
  }
  /**
   * Assumes the given token is a noun.
   * @return true if the token has a synset with an ancestor that is either
   *         Person or Group.
   */
  public boolean isNounPersonOrGroup(String token) {
    return isNounPersonOrGroup(token, false, false);
  }
  public boolean isNounPersonOrGroup(String token, boolean mainSynsetOnly, boolean justPerson) {
    if( _isPersonOrGroup == null ) _isPersonOrGroup = new HashMap<String, Boolean>();
    if( _isPersonOrGroup.containsKey(token) ) return _isPersonOrGroup.get(token);

    Synset[] synsets = synsetsOf(token, POS.NOUN);
    if( synsets == null ) {
    }
    else {
      for( Synset synset : synsets ) {
        List<Synset> chain = hypernymChainKeepChild(synset);

        if( chain != null ) {
          for( Synset parent : chain ) {
            Word[] words = parent.getWords();
            if( (words.length > 0 && words[0].getLemma().equals("person")) ||
                (!justPerson && words.length > 0 && words[0].getLemma().equals("social_group")) ) {
              _isPersonOrGroup.put(token, true);
              return true;
            }
          }
        }
        // Stop now if we are only checking the main synset.
        if( mainSynsetOnly ) return false;
      }
    }
    _isPersonOrGroup.put(token, false);
    return false;
  }

  public boolean isLocation(String token) {
    if( _isLocation == null ) _isLocation = new HashMap<String, Boolean>();
    if( _isLocation.containsKey(token) ) return _isLocation.get(token);

    Synset[] synsets = synsetsOf(token, POS.NOUN);
    //    System.out.println("isNounEntity top " + token);
    if( synsets == null ) {
      //      System.out.println("isNounEntity null synsets: " + token);
    }
    else {
      for( Synset synset : synsets ) {
        List<Synset> chain = hypernymChainKeepChild(synset);
        if( chain != null ) {
          for( Synset parent : chain ) {
            //            System.out.println("\t" + parent);
            Word[] words = parent.getWords();
            if( words.length > 0 &&
                (words[0].getLemma().equals("location") || words[0].getLemma().equals("road")) ) {
              _isLocation.put(token, true);
              return true;
            }
          }
        }
      }
    }
    _isLocation.put(token, false);
    return false;
  }

  /**
   * A more precise lookup of physical structures (more precise than all physical objects)
   */
  public boolean isStructure(String token) {
    if( _isStructure == null ) _isStructure = new HashMap<String, Boolean>();
    if( _isStructure.containsKey(token) ) return _isStructure.get(token);

    Synset[] synsets = synsetsOf(token, POS.NOUN);
    //    System.out.println("isNounEntity top " + token);
    if( synsets == null ) {
      //      System.out.println("isNounEntity null synsets: " + token);
    }
    else {
      for( Synset synset : synsets ) {
        List<Synset> chain = hypernymChainKeepChild(synset);
        if( chain != null ) {
          for( Synset parent : chain ) {
            //            System.out.println("\t" + parent);
            Word[] words = parent.getWords();
            if( words.length > 0 && words[0].getLemma().equals("structure") ) {
              _isStructure.put(token, true);
              return true;
            }
          }
        }
      }
    }
    _isStructure.put(token, false);
    return false;
  }

  /**
   * Assumes the given token is a noun.
   * @return true if the token has a synset with an ancestor that is Integer
   */
  public boolean isInteger(String token) {
    Synset[] synsets = synsetsOf(token, POS.NOUN);
    //    System.out.println("isNounEntity top " + token);
    if( synsets == null ) {
      //      System.out.println("isNounEntity null synsets: " + token);
    }
    else {
      for( Synset synset : synsets ) {
        List<Synset> chain = hypernymChainKeepChild(synset);
        if( chain != null ) {
          for( Synset parent : chain ) {
            //	    System.out.println(parent);
            Word[] words = parent.getWords();
            if( (words.length > 0 && words[0].getLemma().equals("integer")) )
              return true;
          }
        }
      }
    }
    return false;
  }


  /**
   * Assumes the given token is a noun.
   * @return true if the token has a synset with an ancestor that is a physical object
   *         synset.
   */
  public boolean isPhysicalObject(String token) {
    if( _isPhysicalObject == null ) _isPhysicalObject = new HashMap<String, Boolean>();
    if( _isPhysicalObject.containsKey(token) ) return _isPhysicalObject.get(token);

    Synset[] synsets = synsetsOf(token, POS.NOUN);
    //    System.out.println("isPhysicalObject top " + token);
    if( synsets == null ) {
      //      System.out.println("isNounEvent null synsets: " + token);
    }
    else {
      for( Synset synset : synsets ) {
        //        System.out.println("synset = " + synset);
        List<Synset> chain = hypernymChainKeepChild(synset);
        if( chain != null ) {
          for( Synset parent : chain ) {
            //            System.out.println("parent = " + parent);
            Word[] words = parent.getWords();
            if( words.length >= 2 &&
                (words[1].getLemma().equals("physical_object") ||
                    words[0].getLemma().equals("physical_object")) ) {
              _isPhysicalObject.put(token, true);
              return true;
            }
          }
        }
      }
    }
    _isPhysicalObject.put(token, false);
    return false;
  }

  /**
   * Assumes the given token is a noun.
   * WordNet does not put material (explosive, rocks, dust, fiber, etc.) under "physical objects".
   * @return true if the token has a synset with an ancestor that is a "material" synset.
   * 
   */
  public boolean isMaterial(String token) {
    if( _isMaterial == null ) _isMaterial = new HashMap<String, Boolean>();
    if( _isMaterial.containsKey(token) ) return _isMaterial.get(token);

    Synset[] synsets = synsetsOf(token, POS.NOUN);
    //    System.out.println("isMatter top " + token);
    if( synsets == null ) {
    }
    else {
      for( Synset synset : synsets ) {
        List<Synset> chain = hypernymChainKeepChild(synset);
        if( chain != null ) {
          for( Synset parent : chain ) {
            Word[] words = parent.getWords();
            if( words.length >= 1 && words[0].getLemma().equals("material") ) {
              _isMaterial.put(token, true);
              return true;
            }
          }
        }
      }
    }
    _isMaterial.put(token, false);
    return false;
  }

  /**
   * Assumes the given token is a noun.
   * @return true if the token has a synset with an ancestor that is the Event
   *         synset.
   */
  public boolean isMeasure(String token) {
    // save time with a table lookup
    if( _isMeasure == null ) _isMeasure = new HashMap<String, Boolean>();
    if( _isMeasure.containsKey(token) ) return _isMeasure.get(token);

    Synset[] synsets = synsetsOf(token, POS.NOUN);
    //    System.out.println("isPhysicalObject top " + token);
    if( synsets == null ) {
      //      System.out.println("isNounEvent null synsets: " + token);
    }
    else {
      for( Synset synset : synsets ) {
        List<Synset> chain = hypernymChainKeepChild(synset);
        if( chain != null ) {
          for( Synset parent : chain ) {
            //	    System.out.println("parent = " + parent);
            Word[] words = parent.getWords();
            if( words.length >= 1 &&
                words[0].getLemma().equals("measure") ) {
              _isMeasure.put(token, true);
              return true;
            }
          }
        }
      }
    }
    _isMeasure.put(token, false);
    return false;
  }


  /**
   * Assumes the given token is a noun.
   * @return true if the token has a synset with an ancestor that is the Event
   *         synset.
   */
  public boolean isNounEvent(String token) {
    // save time with a table lookup
    if( _isNounEvent == null ) _isNounEvent = new HashMap<String, Boolean>();
    if( _isNounEvent.containsKey(token) ) return _isNounEvent.get(token);

    Synset[] synsets = synsetsOf(token, POS.NOUN);
    //    System.out.println("isNounEvent top " + token);
    if( synsets == null ) {
      //      System.out.println("isNounEvent null synsets: " + token);
    }
    else {
      for( Synset synset : synsets ) {
        List<Synset> chain = hypernymChainKeepChild(synset);
        if( chain != null ) {
          for( Synset parent : chain ) {
            Word[] words = parent.getWords();
            if( words.length == 1 && words[0].getLemma().equals("event") ) {
              _isNounEvent.put(token, true);
              return true;
            }
          }
        }
      }
    }
    _isNounEvent.put(token, false);
    return false;
  }

  public Synset getRootSynset() {
    Synset[] synsets = synsetsOf("entity", POS.NOUN);
    return synsets[0];
  }


  public void test() {
    String word = "explosion";
    String word2 = "kidnapping";
    String word3 = "pipes";
    String word4 = "blown_out";

    System.out.println(word + "  nominal = " + isNominalization(word));
    System.out.println(word2 + " nominal = " + isNominalization(word2));
    System.out.println(word + "  nounevent = " + isNounEvent(word));
    System.out.println(word2 + " nounevent = " + isNounEvent(word2));
    System.out.println(word + " verbs = " + getVerbsOfNominalization(word));
    System.out.println(word2 + " verbs = " + getVerbsOfNominalization(word2));

    System.out.println(word3 + " person = " + isNounPersonOrGroup(word3));

    System.out.println(word4 + " noun lemma = " + nounToLemma(word4));
    System.out.println(word4 + " verb lemma = " + verbToLemma(word4));

    // TEST event words
    String[] words = { "person", "soldier", "army", "group", "people", "brigade", "body", "stream" };
    for( String token : words )
      System.out.println(token + " person = " + isNounPersonOrGroup(token));

    String[] ners = { "santa", "colombia", "george bush", "soldier" };

    for( String ner : ners ) {
      Synset[] synsets = synsetsOf(ner, POS.NOUN);
      for( Synset syn : synsets )
        System.out.println(ner + " syn " + syn);

      Pointer[] links = synsets[0].getPointers();
      for( Pointer link : links ) {
        PointerType type = link.getType();

        System.out.println("  link: " + type + " *label " + type.getLabel() + " *key " + type.getKey());
      }

      List<Synset> h = hypernymChain(synsets[0]);
      System.out.println(ner + " chain: " + h);
    }
  }

  @SuppressWarnings("unchecked")
  public static Map<Synset, Integer> synsetClustering() {
    try {
      Dictionary dict = Dictionary.getInstance();
      Iterator<IndexWord> iterator = dict.getIndexWordIterator(POS.NOUN);

      Map<Synset, Integer> synsetID = new HashMap<Synset, Integer>();
      Map<Integer, Set<Synset>> synsetSet = new HashMap<Integer, Set<Synset>>();

      int maxID = 1;
      // make synset cluster
      while(iterator.hasNext()) {
        IndexWord w = iterator.next();
        List<Integer> seenSynset = new ArrayList<Integer>();
        for(Synset synset : w.getSenses()) {
          if(synsetID.containsKey(synset) && !seenSynset.contains(synsetID.get(synset))) seenSynset.add(synsetID.get(synset));
        }
        int id = -1;
        if(seenSynset.size()==0) {
          // assign new ID for synsets
          id = maxID++;
        } else {
          id = merge(seenSynset, synsetID, synsetSet);
        }
        for(Synset s : w.getSenses()) {
          synsetID.put(s, id);
          if(!synsetSet.containsKey(id)) synsetSet.put(id, new HashSet<Synset>());
          synsetSet.get(id).add(s);
        }
      }
      return synsetID;
    } catch(JWNLException e) {
      throw new RuntimeException(e);
    }
  }

  private static int merge(List<Integer> seenSynset,
      Map<Synset, Integer> synsetID,
      Map<Integer, Set<Synset>> synsetSet) {

    int mergedID = seenSynset.get(0);
    for(int seenID : seenSynset) {
      if (mergedID==seenID) continue;
      for(Synset s : synsetSet.get(seenID)) {
        synsetID.put(s, mergedID);
      }
      synsetSet.get(mergedID).addAll(synsetSet.get(seenID));
      synsetSet.remove(seenID);
    }
    return mergedID;
  }

  public static void main(String[] args) {
    String config = "/u/nlp/data/wordnet/jwnl/jwnl_map_properties.xml";
    WordNetHelper net = new WordNetHelper(config);

    Map<Synset, Integer> map = WordNetHelper.synsetClustering();

    Synset[] ss = net.getNominalizationOfVerb("decide");
    System.err.println();


    //    WordNetHelper net = new WordNetHelper(args[0]);
    //    net.test();
  }
}
