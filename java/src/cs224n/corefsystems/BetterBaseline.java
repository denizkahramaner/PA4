package cs224n.corefsystems;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import cs224n.coref.ClusteredMention;
import cs224n.coref.Document;
import cs224n.coref.Entity;
import cs224n.coref.Mention;
import cs224n.coref.Sentence;
import cs224n.ling.Tree;
import cs224n.util.Pair;

public class BetterBaseline implements CoreferenceSystem {

	@Override
	public void train(Collection<Pair<Document, List<Entity>>> trainingData) {
		// TODO Auto-generated method stub

	}

	@Override
	public List<ClusteredMention> runCoreference(Document doc) {
		// Let's just use exact head match (lowercase)
	    List<ClusteredMention> mentions = new ArrayList<ClusteredMention>();
	    Map<String,Entity> clusters = new HashMap<String,Entity>();
	    //(for each mention...)
	    for(Mention m : doc.getMentions()){
	      //(...get its text)
	      String mentionString = m.headWord().toLowerCase(); // m.gloss();
	      //(...if we've seen this text before...)
	      if(clusters.containsKey(mentionString)){
	        //(...add it to the cluster)
	        mentions.add(m.markCoreferent(clusters.get(mentionString)));
	      } else {
	        //(...else create a new singleton cluster)
	        ClusteredMention newCluster = m.markSingleton();
	        mentions.add(newCluster);
	        clusters.put(mentionString,newCluster.entity);
	      }
	    }
	    //(return the mentions)
	    return mentions;
	}

}
