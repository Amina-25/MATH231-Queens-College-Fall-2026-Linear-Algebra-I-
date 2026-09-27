# hw1.py -- Part F work. Definitions only; nothing runs on import.
import numpy as np

# Data used in F.1
u = np.array([3.0, -4.0])
v = np.array([-1.0, 2.0])
w = np.array([5.0, 0.0])
alpha = -3.0
beta = 0.5

# --- F.1(a) -------------------------------------------------------
def q1a():
    # Agrees with A.2(a).
    return u + v

# --- F.1(b) -------------------------------------------------------
def q1b():
    # Agrees with A.2(b).
    return u - v

# --- F.1(c) -------------------------------------------------------
def q1c():
    # Agrees with A.2(c).
    return v - u

# --- F.1(d) -------------------------------------------------------
def q1d():
    # Agrees with A.2(d).
    return alpha * u

# --- F.1(e) -------------------------------------------------------
def q1e():
    # Agrees with A.2(e).
    return beta * v + u

# --- F.1(f) -------------------------------------------------------
def q1f():
    # Agrees with A.2(f).
    return 2*u - 3*v + w

# --- F.1(g) -------------------------------------------------------
def q1g():
    # Agrees with A.2(g).
    return (np.linalg.norm(u), np.linalg.norm(v), np.linalg.norm(w))

# --- F.1(h) -------------------------------------------------------
def q1h():
    # Agrees with A.2(h).
    return np.linalg.norm(u + v)

# --- F.1(i) -------------------------------------------------------
def q1i():
    # Agrees with A.2(i).
    left = np.linalg.norm(alpha*u)
    right = abs(alpha)*np.linalg.norm(u)
    return (left, right, bool(np.isclose(left, right)))

# --- F.2(a) -------------------------------------------------------
def norm2(vec):
    return np.sqrt(vec[0]**2 + vec[1]**2)

def q2a():
    # norm2 itself is the requested function; return one sample evaluation.
    return norm2(np.array([3.0, 4.0]))

vectors_a1 = [
    np.array([5.0, 12.0]),
    np.array([4.0, 3.0]),
    np.array([4.0, 3.0]),
    np.array([-3.0, 4.0])
]

# --- F.2(b) -------------------------------------------------------
def q2b():
    return [bool(np.isclose(norm2(x), np.linalg.norm(x))) for x in vectors_a1]

# --- F.2(c) -------------------------------------------------------
def q2c():
    norms = [norm2(x) for x in vectors_a1]
    exact_as_decimals = [13.0, 5.0, 5.0, 5.0]
    checks = [bool(np.isclose(a, b)) for a, b in zip(norms, exact_as_decimals)]
    return (norms, checks)

def vector_from(tail, head):
    return head - tail

# --- F.3 ----------------------------------------------------------
def q3():
    tails = [np.array([0.0,0.0]), np.array([2.0,-1.0]),
             np.array([-3.0,4.0]), np.array([1.0,1.0])]
    heads = [np.array([5.0,12.0]), np.array([6.0,2.0]),
             np.array([1.0,7.0]), np.array([-2.0,5.0])]
    results = [vector_from(t, h) for t, h in zip(tails, heads)]
    same = bool(np.allclose(results[1], results[2]))
    return (results, f"rows (b) and (c) are the same vector: {same}")

def triangle_counts():
    rng = np.random.default_rng(231)
    satisfied = 0
    equal = 0
    for _ in range(10000):
        a = rng.normal(size=2)
        b = rng.normal(size=2)
        left = np.linalg.norm(a+b)
        right = np.linalg.norm(a)+np.linalg.norm(b)
        if left <= right or np.isclose(left, right):
            satisfied += 1
        if np.isclose(left, right):
            equal += 1
    return satisfied, equal

# --- F.4(a) -------------------------------------------------------
def q4a():
    return triangle_counts()[0]

# --- F.4(b) -------------------------------------------------------
def q4b():
    return (triangle_counts()[1],
            "np.isclose uses a tolerance, so it can count near-equalities; exact equality for random continuous vectors is overwhelmingly unlikely.")

# --- F.4(c) -------------------------------------------------------
def q4c():
    a = np.array([2.0, 3.0])
    b = np.array([4.0, 6.0])
    left = np.linalg.norm(a+b)
    right = np.linalg.norm(a)+np.linalg.norm(b)
    return (a, b, bool(np.isclose(left, right)),
            "Equality holds because b is a nonnegative scalar multiple of a.")

# --- F.4(d) -------------------------------------------------------
def q4d():
    # Finding no counterexample in 10,000 tests does not prove the theorem.
    # It checks only finitely many pairs, while the theorem is about every pair in R^2.
    return "No. A finite numerical test cannot prove a statement about every pair of vectors."

def cmul(a,b,c,d):
    return complex(a*c-b*d, a*d+b*c)

def cdiv(a,b,c,d):
    denom = c*c+d*d
    if np.isclose(denom, 0.0):
        raise ZeroDivisionError("cannot divide by the zero complex number")
    return complex((a*c+b*d)/denom, (b*c-a*d)/denom)

def cmod(a,b):
    return np.sqrt(a*a+b*b)

# --- F.5(a) -------------------------------------------------------
def q5a():
    z, w0 = 3-2j, -1+4j
    mine = cmul(3.0,-2.0,-1.0,4.0)
    return (mine, z*w0, bool(np.isclose(mine,z*w0)))

# --- F.5(b) -------------------------------------------------------
def q5b():
    z, w0 = 3-2j, -1+4j
    mine = cdiv(3.0,-2.0,-1.0,4.0)
    try:
        cdiv(1.0,2.0,0.0,0.0)
        message = "guard failed"
    except ZeroDivisionError as err:
        message = str(err)
    return (mine, z/w0, bool(np.isclose(mine,z/w0)), message)

# --- F.5(c) -------------------------------------------------------
def q5c():
    z = 3-2j
    mine = cmod(3.0,-2.0)
    return (mine, abs(z), bool(np.isclose(mine,abs(z))))

thetas = np.pi*np.arange(10)/10

# --- F.6(a) -------------------------------------------------------
def q6a():
    return [(float(t), bool(np.isclose(np.exp(1j*t), np.cos(t)+1j*np.sin(t)))) for t in thetas]

# --- F.6(b) -------------------------------------------------------
def q6b():
    discrepancy = np.exp(1j*np.pi)+1
    # Euler's identity gives the exact answer 0. The tiny nonzero imaginary part is floating-point error.
    # Its size, about 10^-16, shows that the discrepancy is near machine precision.
    return (discrepancy, bool(np.isclose(discrepancy,0)))

# --- F.6(c) -------------------------------------------------------
def q6c():
    return bool(np.allclose(np.abs(np.exp(1j*thetas)), np.ones(10)))

# --- F.6(d) -------------------------------------------------------
def q6d():
    nums = [1j, np.sqrt(2)/2+1j*np.sqrt(2)/2,
            0.5+1j*np.sqrt(3)/2, -3/5+1j*4/5]
    expected = [np.pi/2, np.pi/4, np.pi/3, np.arctan2(4,-3)]
    return [(float(np.angle(z)), float(t), bool(np.isclose(np.angle(z),t)))
            for z,t in zip(nums,expected)]

# --- F.7(a) -------------------------------------------------------
def q7a():
    z=3+2j
    values=[(1j**k)*z for k in range(5)]
    return (values, bool(np.isclose(values[4],values[0])))

# --- F.7(b) -------------------------------------------------------
def q7b():
    rng=np.random.default_rng(231)
    zs=rng.normal(size=1000)+1j*rng.normal(size=1000)
    return bool(np.allclose(np.abs(1j*zs),np.abs(zs)))

# --- F.7(c) -------------------------------------------------------
def q7c():
    corners=np.array([0+0j,1+0j,1+1j,0+1j])
    once=1j*corners
    four_times=(1j**4)*corners
    return (once, four_times, bool(np.allclose(four_times,corners)))

# --- F.7(d) -------------------------------------------------------
def q7d():
    z=4+1j
    r90=z*1j
    r180=z*(-1)
    r30=z*np.exp(1j*np.pi/6)
    return [(float(x.real),float(x.imag)) for x in (r90,r180,r30)]

# --- F.8 ----------------------------------------------------------
def q8():
    z, w0 = 3-2j, -1+4j
    c2 = {
        "C.2(a)": (z+w0,z-w0,w0-z), "C.2(b)": z*w0, "C.2(c)": z**2,
        "C.2(d)": (z.conjugate(),z*z.conjugate()), "C.2(e)": z/w0,
        "C.2(f)": 1/w0, "C.2(g)": (abs(z),abs(w0),abs(z*w0),abs(z)*abs(w0))}
    z4=5-12j
    c4={"C.4(a)":(z4.conjugate(),z4*z4.conjugate(),abs(z4),abs(z4)**2),
        "C.4(b)":(z4+z4.conjugate(),z4-z4.conjugate())}
    c5={"C.5(a)":(abs(-3+4j),abs(1+1j),abs(-7+0j),abs(2j),abs(0j)),
        "C.5(b)":(abs((3+2j)-(-1-1j)),abs((5-1j)-(5+3j)))}
    # The numerical results agree with the hand computations in C.2, C.4(a)-(b), and C.5(a)-(b).
    # NumPy can check computational problems in Parts A and C and E.1, E.3, E.4 on specific inputs.
    # It cannot prove universal claims such as B.3, D.3, or E.2 (Cauchy-Schwarz): a run can test examples,
    # but even many successful examples do not establish that the statement holds for every allowed vector.
    return (c2,c4,c5)
