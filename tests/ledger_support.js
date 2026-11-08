'use strict';
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
function equal(a,b,p='root'){if(typeof a==='number'&&typeof b==='number'){assert.ok(Number.isFinite(a)&&Number.isFinite(b),p);if(Number.isInteger(a)&&Number.isInteger(b))return assert.equal(a,b,p);assert.ok(Math.abs(a-b)<=1e-12+1e-12*Math.abs(b),p+':'+a+' vs '+b);return;}if(a===null||b===null||typeof a!=='object'||typeof b!=='object')return assert.equal(a,b,p);assert.equal(Array.isArray(a),Array.isArray(b),p);assert.deepEqual(Object.keys(a).sort(),Object.keys(b).sort(),p);for(const k of Object.keys(a))equal(a[k],b[k],p+'.'+k);}
const root=path.join(__dirname,'../setguard/artifacts');
const ws=fs.readdirSync(root).filter(x=>/^rank-\d+\.json$/.test(x)).sort().flatMap(x=>JSON.parse(fs.readFileSync(path.join(root,x))));
module.exports={equal,ws};
