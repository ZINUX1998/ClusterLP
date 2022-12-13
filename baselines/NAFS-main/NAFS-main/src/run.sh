echo "Clustering_simple results: "
python clustering_simple.py --dataset=cora --hops=8
python clustering_simple.py --dataset=citeseer --hops=8
python clustering_simple.py --dataset=pubmed --hops=8
python clustering_simple.py --dataset=wiki --hops=2
echo "Clustering_mean results: "
python clustering_mean.py --dataset=cora --hops=17
python clustering_mean.py --dataset=citeseer --hops=15
python clustering_mean.py --dataset=pubmed --hops=56
python clustering_mean.py --dataset=wiki --hops=8
echo "Clustering_max results: "
python clustering_max.py --dataset=cora --hops=10
python clustering_max.py --dataset=citeseer --hops=10
python clustering_max.py --dataset=pubmed --hops=65
python clustering_max.py --dataset=wiki --hops=10
echo "Clustering_concat results: "
python clustering_concat.py --dataset=cora --hops=19
python clustering_concat.py --dataset=citeseer --hops=10
python clustering_concat.py --dataset=pubmed --hops=55
python clustering_concat.py --dataset=wiki --hops=7
echo "Link_prediction_simple results: "
python link_prediction_simple.py --dataset=cora --hops=2
python link_prediction_simple.py --dataset=citeseer --hops=2
python link_prediction_simple.py --dataset=pubmed --hops=2
echo "Link_prediction_mean results: "
python link_prediction_mean.py --dataset=cora --hops=9
python link_prediction_mean.py --dataset=citeseer --hops=4
python link_prediction_mean.py --dataset=pubmed --hops=6
echo "Link_prediction_max results: "
python link_prediction_max.py --dataset=cora --hops=8
python link_prediction_max.py --dataset=citeseer --hops=4
python link_prediction_max.py --dataset=pubmed --hops=5
echo "Link_prediction_concat results: "
python link_prediction_concat.py --dataset=cora --hops=10
python link_prediction_concat.py --dataset=citeseer --hops=12
python link_prediction_concat.py --dataset=pubmed --hops=6
