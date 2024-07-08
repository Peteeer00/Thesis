import pandas as pd
import numpy as np
import statsmodels.api as sm
        
def PCA(stock, data, plot):
        if plot == True:
                from plotly.graph_objs import Bar, Scatter, Layout, Figure
                from plotly.graph_objs.layout import YAxis
                from plotly.offline import iplot
                import matplotlib.pyplot as plt
        
        SP500moves = data
        aaplmoves = stock
        SP500norm = (SP500moves - SP500moves.mean()) / SP500moves.std()
        aaplnorm = (aaplmoves - aaplmoves.mean()) / aaplmoves.std()
        cov = SP500norm.cov()
        eig_vals, eig_vecs = np.linalg.eig(cov)
        eig_vals = np.real(eig_vals)
        eig_vecs = np.real(eig_vecs)
        tot = sum(eig_vals)
        var_exp = [(i / tot) * 100 for i in sorted(eig_vals, reverse=True)]
        cum_var_exp = np.cumsum(var_exp)

        if plot == True:
                trace1 = Bar(
                        x=['PC %s' %i for i in range(1, 100)],
                        y=var_exp,
                        showlegend=False)

                trace2 = Scatter(
                        x=['PC %s' %i for i in range(1, 100)], 
                        y=cum_var_exp,
                        name='Cumulative explained variance')

                data = (trace1, trace2)

                layout = Layout(
                        yaxis=YAxis(title='Explained variance in percent'),
                        title='Explained variance per principal component')

                fig = Figure(data=data, layout=layout)
                from plotly.offline import iplot
                iplot(fig)



        PCs = np.matmul(data.values, eig_vecs)
        PCs = pd.DataFrame(PCs, index=data.index)
        PCs['AAPL'] = aaplmoves

        colnames = ['PC'+str(i) for i in np.arange(len(PCs.columns.values) - 1)] + ['AAPL']
        PCs.columns = colnames
        PC10 = PCs[['PC0','PC1','PC2','PC3','PC4','PC5','PC6','PC7','PC8','PC9','PC10','AAPL']]
        if plot == True:
                PC10.plot(figsize=(16, 10))
                plt.ylabel('Normalised Values');


        Y = PC10['AAPL']
        X = PC10[['PC0','PC1','PC2','PC3','PC4','PC5','PC6','PC7', 'PC8','PC9']]
        X = sm.add_constant(X)
        model = sm.OLS(Y, X)
        results = model.fit()


        results.summary()

        return cum_var_exp, PC10, results


