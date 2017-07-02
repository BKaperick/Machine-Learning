
class Cost():
    def __init__(self):
        pass

    def cost(self, output_emp, output_act):
        pass

    def gradient(self, output_emp, output_act):
        pass

class DiscreteClassifyError(Cost):

    def cost(output_emp, output_act):
        ''' Count of misclassifications between empirical and actual outputs '''
        correct_count = sum(output_emp.argmax(0) == output_act.argmax(0))
        return correct_count

class MeanSquaredError(Cost):

    def cost(output_emp, output_act):
        ''' Mean squared error between empirical and actual outputs '''
        cost_val = .5 * sum(np.linalg.norm(output_emp - output_act, axis=0) ** 2) / n
        return cost_val

    def gradient(output_emp, output_act):
        ''' Gradient for Quadratic cost function defined in cost() '''
        return output_emp - output_act


class CrossEntropyError(Cost):

    def cost(output_emp, output_act):
        ''' Cross entropy error between empirical and actual outputs '''
        n = output_emp.shape[1]
        cost_val = 0
        
        for i,a in enumerate(output_emp.T):
            y = output_act[:,i]
            term = sum( y * np.log(a) + (1-y) * np.log(1-a) )
            cost_val += term
        # Standard normalization factor
        cost_val *= (-1/n)
        return cost_val

    def gradient(output_emp, output_act):
        ''' Gradient for cross-entropy cost function defined in cost() '''
        return output_emp - output_act


