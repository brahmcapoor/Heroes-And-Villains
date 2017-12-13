import dill
from sklearn.cluster import KMeans
import numpy as np
from constants import *
import matplotlib as mpl 
mpl.use('TkAgg')
import matplotlib.pyplot as plt




def sentiment_clustering():
	X = np.array
	avg_sentiments = None
	with open('avg_sentiment.pkl', 'rb') as f:
		avg_sentiments = dill.load(f)
		f.close()

	protag_sentiments = None
	with open('protagonist_sentiment.pkl', 'rb') as f:
		protag_sentiments = dill.load(f)
		f.close()

	movieids_to_names = None
	with open('movieids_to_names.pkl', 'rb') as f:
		movieids_to_names = dill.load(f)
		f.close()

	charids_to_names = None
	with open('charids_to_names.pkl', 'rb') as f:
		charids_to_names = dill.load(f)
		f.close()

	
	with open(LABELLED_DATA_FILENAME, 'r', encoding=FILE_ENCODING) as dataset:
		for movie in dataset:
			char_sentiments = []
			movie = movie.strip().split(SEPARATOR)
			m_id = movie[0]
			cur_avg_sentiments = avg_sentiments[m_id]
			cur_protag_sentiments = protag_sentiments[m_id]
			sentiments_to_charids = {}
			for k,v in cur_avg_sentiments.items():
				cluster_val = [v[0],cur_protag_sentiments[k][0]]
				char_sentiments.append(cluster_val)
				sentiments_to_charids[tuple(cluster_val)] = k

			X = np.array(char_sentiments)

			kmeans = KMeans(n_clusters=3).fit(X)
			print(kmeans.cluster_centers_)

			# Step size of the mesh. Decrease to increase the quality of the VQ.
			h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].

			# Plot the decision boundary. For that, we will assign a color to each
			x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
			y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
			xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

			# Obtain labels for each point in mesh. Use last trained model.
			Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
			print(X)

			# Put the result into a color plot
			Z = Z.reshape(xx.shape)
			plt.figure(1)
			plt.clf()
			plt.imshow(Z, interpolation='nearest',
			           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
			           cmap=plt.cm.Paired,
			           aspect='auto', origin='lower')
			plt.xlabel('average sentiment')
			plt.ylabel('protagonist sentiment towards')

			#plt.plot(X[:, 0], X[:, 1], 'k.', markersize=1)
			colors = ['k', 'g', 'r', 'c', 'm', 'y', 'b',]
			styles = ['D', 'o', 'v', '*', '+', '3', '4']
			for i in range(len(X)):
				cur_label = charids_to_names[sentiments_to_charids[(X[i,0], X[i,1])], m_id]
				plt.plot(X[i,0], X[i,1], styles[i//7]+colors[i%7], markersize=5, label=cur_label)
				#plt.annotate(i, xy = (X[i,0], X[i,1]), xytext = (0, 0), textcoords = 'offset points')
			#for label in zip(labels):

				#plt.annotate(label, (elem[0], elem[1]))
			# Plot the centroids as a white X
			plt.legend(bbox_to_anchor=(.97,1))
			centroids = kmeans.cluster_centers_
			#plt.scatter(centroids[:, 0], centroids[:, 1],
			            #marker='x', s=169, linewidths=3,
			            #color='w', zorder=10)
			plt.title('3-means clustering on {}'.format(movieids_to_names[m_id]))
			plt.xlim(x_min, x_max)
			plt.ylim(y_min, y_max)
			plt.xticks(())
			plt.yticks(())
			plt.show()

			





	


	

sentiment_clustering()