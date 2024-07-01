void jacobistep(int m, int n, double psinew[m+2][n+2], double psi[m+2][n+2]);

void jacobistepvort(int m, int n,
                    double zetnew[m+2][n+2], double psinew[m+2][n+2],
		    double zet[m+2][n+2],    double psi[m+2][n+2],
		    double re);
double deltasq(int m, int n, double newarr[m+2][n+2], double oldarr[m+2][n+2]);
