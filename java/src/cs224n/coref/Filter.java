package cs224n.coref;

import java.util.*;
public interface Filter{
	
	public Map<Mention,HashSet<Mention>> filter(Map<Mention,HashSet<Mention>> mentions,Document doc);
	
}